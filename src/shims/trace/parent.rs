use std::sync::atomic::{AtomicPtr, AtomicUsize};

use ipc_channel::ipc;
use nix::sys::{ptrace, signal, wait};
use nix::unistd;

use crate::shims::trace::{
    AccessEvent, FAKE_STACK_SIZE, LibcEvent, MemEvents, MmapEvent, StartFfiInfo, TraceRequest,
};

/// The flags to use when calling `waitid()`.
/// Since bitwise or on the nix version of these flags is implemented as a trait,
/// this cannot be const directly so we do it this way.
const WAIT_FLAGS: wait::WaitPidFlag =
    wait::WaitPidFlag::from_bits_truncate(libc::WUNTRACED | libc::WEXITED);

/// Opcode for an instruction to raise SIGTRAP, to be written in the child process.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const BREAKPT_INSTR: isize = 0xCC;
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
const BREAKPT_INSTR: isize = 0xD420;
// FIXME: riscv!

/// The size of the breakpoint-triggering instruction, in bytes.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const BREAKPT_INSTR_SIZE: usize = 1;
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
const BREAKPT_INSTR_SIZE: usize = 4;

/// Arch-specific maximum size a single access might perform. x86 value is set
/// assuming nothing bigger than AVX-512 is available.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const ARCH_MAX_ACCESS_SIZE: usize = 64;
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
const ARCH_MAX_ACCESS_SIZE: usize = 16;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
const ARCH_MAX_ACCESS_SIZE: usize = 16;

/// The default word size on a given platform, in bytes.
#[cfg(any(target_arch = "x86", target_arch = "arm", target_arch = "riscv32"))]
const ARCH_WORD_SIZE: usize = 4;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64"))]
const ARCH_WORD_SIZE: usize = 8;

/// The address of the page set to be edited, initialised to a sentinel null
/// pointer.
static PAGE_ADDR: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());
/// The host pagesize, initialised to a sentinel zero value.
pub static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);
/// How many consecutive pages to unprotect. 1 by default, unlikely to be set
/// higher than 2.
static PAGE_COUNT: AtomicUsize = AtomicUsize::new(1);

/// Allows us to get common arguments from the `user_regs_t` across architectures.
/// Normally this would land us ABI hell, but thankfully all of our usecases
/// consist of functions with a small number of register-sized integer arguments.
/// See <https://man7.org/linux/man-pages/man2/syscall.2.html> for sources
trait ArchIndependentRegs {
    /// The return value of a function call, if one just happened. All of our
    /// uses of it involve this being signed, so return it as such.
    fn retval(&self) -> isize;
    /// The first ptr-sized argument.
    fn arg1(&self) -> usize;
    /// The second ptr-sized argument.
    fn arg2(&self) -> usize;
    /// If entering a syscall, this is the syscall number. `libc` has this as
    /// a signed integer, so return it that way here also.
    fn syscall_nr(&self) -> isize;
    /// The instruction pointer.
    fn ip(&self) -> usize;
    /// The stack pointer.
    fn sp(&self) -> usize;
    /// Set the instruction pointer; remember to also set the stack pointer, or
    /// else the stack might get messed up!
    fn set_ip(&mut self, ip: usize);
    /// Set the stack pointer, ideally to a zeroed-out area.
    fn set_sp(&mut self, sp: usize);
}

// It's fine / desirable behaviour for values to wrap here, we care about just
// preserving the bit pattern
#[cfg(target_arch = "x86_64")]
#[expect(clippy::as_conversions)]
#[rustfmt::skip]
impl ArchIndependentRegs for libc::user_regs_struct {
    fn retval(&self) -> isize { self.rax as _ }
    fn arg1(&self) -> usize { self.rdi as _ }
    fn arg2(&self) -> usize { self.rsi as _ }
    fn syscall_nr(&self) -> isize { self.orig_rax as _ }
    fn ip(&self) -> usize { self.rip as _ }
    fn sp(&self) -> usize { self.rsp as _ }
    fn set_ip(&mut self, ip: usize) { self.rip = ip as _ }
    fn set_sp(&mut self, sp: usize) { self.rsp = sp as _ }
}

#[cfg(target_arch = "x86")]
#[expect(clippy::as_conversions)]
#[rustfmt::skip]
impl ArchIndependentRegs for libc::user_regs_struct {
    fn retval(&self) -> isize { self.eax as _ }
    fn arg1(&self) -> usize { self.edi as _ }
    fn arg2(&self) -> usize { self.esi as _ }
    fn syscall_nr(&self) -> isize { self.orig_eax as _ }
    fn ip(&self) -> usize { self.eip as _ }
    fn sp(&self) -> usize { self.esp as _ }
    fn set_ip(&mut self, ip: usize) { self.eip = ip as _ }
    fn set_sp(&mut self, sp: usize) { self.esp = sp as _ }
}

#[cfg(target_arch = "aarch64")]
#[expect(clippy::as_conversions)]
#[rustfmt::skip]
impl ArchIndependentRegs for libc::user_regs_struct {
    fn retval(&self) -> isize { self.regs[0] as _ }
    fn arg1(&self) -> usize { self.regs[0] as _ }
    fn arg2(&self) -> usize { self.regs[1] as _ }
    fn syscall_nr(&self) -> isize { self.regs[8] as _ }
    fn ip(&self) -> usize { self.pc as _ }
    fn sp(&self) -> usize { self.sp as _ }
    fn set_ip(&mut self, ip: usize) { self.pc = ip as _ }
    fn set_sp(&mut self, sp: usize) { self.sp = sp as _ }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
#[expect(clippy::as_conversions)]
#[rustfmt::skip]
impl ArchIndependentRegs for libc::user_regs_struct {
    fn ip(&self) -> usize { self.pc as _ }
    fn set_ip(&mut self, ip: usize) { self.pc = ip as _ }
    fn set_sp(&mut self, sp: usize) { self.sp = sp as _ }
}

/// A unified event representing something happening on the child process. Wraps
/// `nix`'s `WaitStatus` and our custom signals so it can all be done with one
/// `match` statement.
pub enum ExecEvent {
    /// Child process requests that we begin monitoring it.
    Start(StartFfiInfo),
    /// Child requests that we stop monitoring and pass over the events we
    /// detected.
    End,
    /// The child process with the specified pid was stopped by the given signal.
    Status(unistd::Pid, signal::Signal),
    /// The child process with the specified pid entered or existed a syscall.
    Syscall(unistd::Pid),
    /// A child process exited or was killed; if we have a return code, it is
    /// specified.
    Died(Option<i32>),
}

/// A listener for the FFI start info channel along with relevant state.
pub struct ChildListener {
    /// The matching channel for the child's `Supervisor` struct.
    pub message_rx: ipc::IpcReceiver<TraceRequest>,
    /// The main child process' pid.
    pub pid: unistd::Pid,
    /// Whether an FFI call is currently ongoing.
    pub attached: bool,
    /// If `Some`, overrides the return code with the given value.
    pub override_retcode: Option<i32>,
}

impl Iterator for ChildListener {
    type Item = ExecEvent;

    // Allows us to monitor the child process by just iterating over the listener
    // NB: This should never return None!
    fn next(&mut self) -> Option<Self::Item> {
        // Do not block if the child has nothing to report for `waitid`
        let opts = WAIT_FLAGS | wait::WaitPidFlag::WNOHANG;
        loop {
            // Listen to any child, not just the main one. Important if we want
            // to allow the C code to fork further, along with being a bit of
            // defensive programming since Linux sometimes assigns threads of
            // the same process different PIDs with unpredictable rules...
            match wait::waitid(wait::Id::All, opts) {
                Ok(stat) =>
                    match stat {
                        // Child exited normally with a specific code set
                        wait::WaitStatus::Exited(_, code) => {
                            //eprintln!("Exited main {code}");
                            let code = self.override_retcode.unwrap_or(code);
                            return Some(ExecEvent::Died(Some(code)));
                        }
                        // Child was killed by a signal, without giving a code
                        wait::WaitStatus::Signaled(_, _, _) =>
                            return Some(ExecEvent::Died(self.override_retcode)),
                        // Child entered a syscall. Since we're always technically
                        // tracing, only pass this along if we're actively
                        // monitoring the child
                        wait::WaitStatus::PtraceSyscall(pid) =>
                            if self.attached {
                                return Some(ExecEvent::Syscall(pid));
                            },
                        // Child with the given pid was stopped by the given signal.
                        // It's somewhat dubious when this is returned instead of
                        // WaitStatus::Stopped, but for our purposes they are the
                        // same thing.
                        wait::WaitStatus::PtraceEvent(pid, signal, _) =>
                            if self.attached {
                                // This is our end-of-FFI signal!
                                if signal == signal::SIGUSR1 {
                                    self.attached = false;
                                    return Some(ExecEvent::End);
                                } else {
                                    return Some(ExecEvent::Status(pid, signal));
                                }
                            } else {
                                // Log that this happened and pass along the signal.
                                // If we don't do the kill, the child will instead
                                // act as if it never received this signal!
                                eprintln!("Ignoring PtraceEvent {signal:?}");
                                signal::kill(pid, signal).unwrap();
                            },
                        // Child was stopped at the given signal. Same logic as for
                        // WaitStatus::PtraceEvent
                        wait::WaitStatus::Stopped(pid, signal) =>
                            if self.attached {
                                if signal == signal::SIGUSR1 {
                                    self.attached = false;
                                    return Some(ExecEvent::End);
                                } else {
                                    return Some(ExecEvent::Status(pid, signal));
                                }
                            } else {
                                eprintln!("Ignoring Stopped {signal:?}");
                                signal::kill(pid, signal).unwrap();
                            },
                        _ => (),
                    },
                // This case should only trigger if all children died and we
                // somehow missed that, but it's best we not allow any room
                // for deadlocks
                Err(_) => return Some(ExecEvent::Died(None)),
            }

            // Similarly, do a non-blocking poll of the IPC channel
            if let Ok(req) = self.message_rx.try_recv() {
                match req {
                    TraceRequest::StartFfi(info) =>
                    // Should never trigger - but better to panic explicitly than deadlock!
                        if self.attached {
                            panic!("Attempting to begin FFI multiple times!");
                        } else {
                            self.attached = true;
                            return Some(ExecEvent::Start(info));
                        },
                    TraceRequest::OverrideRetcode(code) => self.override_retcode = Some(code),
                }
            }

            // Not ideal, but doing anything else might sacrifice performance
            std::thread::yield_now();
        }
    }
}

/// Values needed for us to intercept calls to certain libc functions, such as
/// their addresses and original content (before we overwrote them).
#[derive(Clone, Copy)]
struct MagicLibcValues {
    /// The address at which `libc::malloc()` begins in memory.
    malloc_addr: usize,
    /// The address at which `libc::realloc()` begins in memory.
    realloc_addr: usize,
    /// The address at which `libc::free()` begins in memory.
    free_addr: usize,
    /// The first word of machine code in `libc::malloc()`.
    malloc_bytes: isize,
    /// The first word of machine code in `libc::realloc()`.
    free_bytes: isize,
    /// The first word of machine code in `libc::free()`.
    realloc_bytes: isize,
}

impl MagicLibcValues {
    /// Gets the needed values. Note that while safe to do after, this should
    /// be done *before* anything is overwritten with `raise(SIGTRAP)`s.
    /// 
    /// While `ptrace::{read, write}` say they need an i64/i32, it's actually
    /// just an `isize` since it depends on the platform.
    #[expect(clippy::as_conversions)]
    fn read() -> Self {
        // No other real way to do this
        let malloc_addr = libc::malloc as usize;
        let realloc_addr = libc::realloc as usize;
        let free_addr = libc::free as usize;
        Self {
            malloc_addr,
            realloc_addr,
            free_addr,
            // I'm sorry...
            // SAFETY: These are all functions that are known to exist if libc
            // is linked against, and they are larger than 8 bytes.
            malloc_bytes: unsafe {
                std::ptr::with_exposed_provenance::<isize>(malloc_addr).read_volatile()
            },
            realloc_bytes: unsafe {
                std::ptr::with_exposed_provenance::<isize>(realloc_addr).read_volatile()
            },
            free_bytes: unsafe {
                std::ptr::with_exposed_provenance::<isize>(free_addr).read_volatile()
            },
        }
    }

    /// Restores the data at the start of the stored functions to its original
    /// contents on the child process.
    #[expect(clippy::as_conversions)]
    fn restore(&self, pid: unistd::Pid) -> Result<(), nix::errno::Errno> {
        ptrace::write(
            pid,
            std::ptr::with_exposed_provenance_mut(self.malloc_addr),
            self.malloc_bytes as _,
        )?;
        ptrace::write(
            pid,
            std::ptr::with_exposed_provenance_mut(self.realloc_addr),
            self.realloc_bytes as _,
        )?;
        ptrace::write(pid, std::ptr::with_exposed_provenance_mut(self.free_addr), self.free_bytes as _)
    }

    /// Overwrites the first 8 bytes of the `_addr` fields with the specified
    /// data, in the child process. `data` should probably be some kind of
    /// breakpoint/SIGTRAP instruction.
    #[expect(clippy::as_conversions)]
    fn overwrite_all(&self, pid: unistd::Pid, data: isize) -> Result<(), nix::errno::Errno> {
        ptrace::write(pid, std::ptr::with_exposed_provenance_mut(self.malloc_addr), data as _)?;
        ptrace::write(pid, std::ptr::with_exposed_provenance_mut(self.realloc_addr), data as _)?;
        ptrace::write(pid, std::ptr::with_exposed_provenance_mut(self.free_addr), data as _)
    }
}

/// An error came up while waiting on the child process to do something.
#[derive(Debug)]
enum ExecError {
    /// The child process died with this return code, if we have one.
    Died(Option<i32>),
    /// Something errored, but we should ignore it and proceed.
    Shrug,
}

/// This is the main loop of the supervisor process. It runs in a separate
/// process from the rest of Miri (but because we fork, addresses for anything
/// created before the fork - like statics - are the same).
pub fn sv_loop(
    listener: ChildListener,
    event_tx: ipc::IpcSender<MemEvents>,
    confirm_tx: ipc::IpcSender<()>,
    page_size: usize,
) -> Result<!, Option<i32>> {
    // Things that we return to the child process
    let mut acc_events = Vec::new();
    let mut mmap_events = Vec::new();
    let mut libc_events = Vec::new();

    // Memory allocated on the MiriMachine
    let mut ch_pages = Vec::new();
    let mut ch_stack = None;

    // Bits needed to intercept libc calls that we care about
    let libc_vals = MagicLibcValues::read();

    // An instance of the Capstone disassembler, so we don't spawn one on every access
    let cs = get_disasm();

    // The pid of the process that we forked from, used by default if we don't
    // have a reason to use another one.
    let main_pid = listener.pid;

    // There's an initial sigstop we need to deal with
    wait_for_signal(main_pid, signal::SIGSTOP, false).map_err(|e| {
        match e {
            ExecError::Died(code) => code,
            ExecError::Shrug => None,
        }
    })?;
    ptrace::cont(main_pid, None).unwrap();

    for evt in listener {
        match evt {
            // start_ffi was called by the child, so prep memory
            ExecEvent::Start(ch_info) => {
                // All the pages that the child process is "allowed to" access
                ch_pages = ch_info.page_ptrs;
                // And the fake stack it allocated for us to use later
                ch_stack = Some(ch_info.stack_ptr);

                // We received the signal and are no longer in the main listener loop,
                // so we can let the child move on to the end of start_ffi where it will
                // raise a SIGSTOP. We need it to be signal-stopped *and waited for* in
                // order to do most ptrace operations!
                confirm_tx.send(()).unwrap();
                wait_for_signal(main_pid, signal::SIGSTOP, false).unwrap();

                // Now overwrite the libc bits we care about monitoring, and tell the child
                // to continue (and begin the real FFI call)
                libc_vals.overwrite_all(main_pid, BREAKPT_INSTR).unwrap();
                ptrace::syscall(main_pid, None).unwrap();
            }
            // end_ffi was called by the child
            ExecEvent::End => {
                // Hand over the access info we traced
                event_tx
                    .send(MemEvents {
                        acc_events,
                        alloc_cutoff: page_size,
                        mmap_events,
                        libc_events,
                    })
                    .unwrap();
                // And reset our values
                acc_events = Vec::new();
                mmap_events = Vec::new();
                libc_events = Vec::new();
                ch_stack = None;

                // Child is already stopped, since it raised SIGUSR1, so we don't
                // need to wait on anything
                libc_vals.restore(main_pid).unwrap();
                // No need to monitor syscalls anymore, they'd just be ignored
                ptrace::cont(main_pid, None).unwrap();
            }
            // Child process was stopped by a signal
            ExecEvent::Status(pid, signal) =>
                match signal {
                    // If it was a segfault, check if it was an artificial one
                    // caused by it trying to access the MiriMachine memory
                    signal::SIGSEGV =>
                        match handle_segfault(
                            pid,
                            &ch_pages,
                            ch_stack.unwrap(),
                            page_size,
                            &cs,
                            &mut acc_events,
                        ) {
                            Err(e) =>
                                match e {
                                    ExecError::Died(code) => return Err(code),
                                    ExecError::Shrug => continue,
                                },
                            _ => (),
                        },
                    // Most likely triggered by the child touching the libc bits
                    // we made trap, so handle that
                    signal::SIGTRAP =>
                        match handle_sigtrap(pid, &mut libc_events, libc_vals) {
                            Err(e) =>
                                match e {
                                    ExecError::Died(code) => return Err(code),
                                    ExecError::Shrug => continue,
                                },
                            _ => (),
                        },
                    // Something weird happened
                    _ => {
                        eprintln!("Process unexpectedly got {signal}; continuing...");
                        // In case we're not tracing
                        if ptrace::syscall(pid, None).is_err() {
                            // If *this* fails too, something really weird happened
                            // and it's probably best to just panic
                            signal::kill(pid, signal::SIGCONT).unwrap();
                        }
                    }
                },
            // Child entered a syscall; we wait for exits inside of this, so it
            // should never trigger on return from a syscall we care about
            ExecEvent::Syscall(pid) => {
                let regs = ptrace::getregs(pid).unwrap();
                // Again, the constants are defined as i64/i32 but they're just isizes
                #[expect(clippy::as_conversions)]
                match regs.syscall_nr() as _ {
                    libc::SYS_mmap => {
                        // No need for a discrete fn here, it's very tiny.
                        // The length is guaranteed to be to be a multiple of
                        // the pagesize anyways so we can just assume it and
                        // the syscall will error if it's not
                        let pg_count = regs.arg2().strict_div(page_size);
                        // Wait for the exit from the call now
                        let regs = match wait_for_syscall(pid, libc::SYS_mmap as _) {
                            Ok(regs) => regs,
                            Err(e) =>
                                match e {
                                    ExecError::Died(code) => return Err(code),
                                    ExecError::Shrug => continue,
                                },
                        };
                        // For mmap, a negative retval is failure, and anything
                        // else is the address it returned
                        if let Ok(addr) = usize::try_from(regs.retval()) {
                            // NB: Don't get rid of munmaps that might have happened,
                            // since pointers to deallocated memory that "by chance"
                            // gets reallocated should get invalidated!
                            for i in 0..pg_count {
                                mmap_events.push(MmapEvent::Mmap(
                                    addr.strict_add(i.strict_mul(page_size)),
                                ));
                            }
                        }
                    }
                    libc::SYS_munmap => {
                        // Register unmapping, or remove a mapping. If a mapping
                        // was entirely transient (i.e. appeared and was deleted
                        // during the tracing), no need to report it
                        match handle_munmap(pid, regs, &mut mmap_events, page_size) {
                            Err(e) =>
                                match e {
                                    ExecError::Died(code) => return Err(code),
                                    ExecError::Shrug => continue,
                                },
                            _ => (),
                        }
                    }
                    // TODO: handle brk/sbrk
                    // or not, using sbrk in 2025 means you deserve UB
                    // Also maybe intercept/prevent fork() et al.?
                    _ => (),
                }

                ptrace::syscall(pid, None).unwrap();
            }
            ExecEvent::Died(code) => {
                return Err(code);
            }
        }
    }

    unreachable!()
}

/// Spawns a Capstone disassembler for the host architecture.
#[rustfmt::skip]
fn get_disasm() -> capstone::Capstone {
    use capstone::prelude::*;
    let cs_pre = Capstone::new();
    {
        #[cfg(target_arch = "x86_64")]
        {cs_pre.x86().mode(arch::x86::ArchMode::Mode64)}
        #[cfg(target_arch = "x86")]
        {cs_pre.x86().mode(arch::x86::ArchMode::Mode32)}
        #[cfg(target_arch = "aarch64")]
        {cs_pre.arm64()}
        #[cfg(target_arch = "arm")]
        {cs_pre.arm()}
        #[cfg(target_arch = "riscv64")]
        {cs_pre.riscv().mode(arch::riscv::ArchMode::RiscV64)}
        #[cfg(target_arch = "riscv32")]
        {cs_pre.riscv().mode(arch::riscv::ArchMode::RiscV32)}
    }
    .detail(true)
    .build()
    .unwrap()
}

/// Waits for `wait_signal`. If `init_cont`, it will first do a `ptrace::cont`.
/// We want to avoid that in some cases, like at the beginning of FFI.
fn wait_for_signal(
    pid: unistd::Pid,
    wait_signal: signal::Signal,
    init_cont: bool,
) -> Result<(), ExecError> {
    if init_cont {
        ptrace::cont(pid, None).unwrap();
    }
    // Repeatedly call `waitid` until we get the signal we want, or the process dies
    loop {
        let stat =
            wait::waitid(wait::Id::Pid(pid), WAIT_FLAGS).map_err(|_| ExecError::Died(None))?;
        let signal = match stat {
            // Report the cause of death, if we know it
            wait::WaitStatus::Exited(_, code) => {
                //eprintln!("Exited sig1 {code}");
                return Err(ExecError::Died(Some(code)));
            }
            wait::WaitStatus::Signaled(_, _, _) => return Err(ExecError::Died(None)),
            wait::WaitStatus::Stopped(_, signal) => signal,
            wait::WaitStatus::PtraceEvent(_, signal, _) => signal,
            // This covers PtraceSyscall and variants that are impossible with
            // the flags set (e.g. WaitStatus::StillAlive)
            _ => {
                ptrace::cont(pid, None).unwrap();
                continue;
            }
        };
        if signal == wait_signal {
            break;
        } else {
            ptrace::cont(pid, None).map_err(|_| ExecError::Died(None))?;
        }
    }
    Ok(())
}

/// Waits for the child to return from its current syscall, grabbing its registers.
/// DO NOT call `ptrace::syscall()` right before this!
fn wait_for_syscall(pid: unistd::Pid, syscall: isize) -> Result<libc::user_regs_struct, ExecError> {
    // We always want an initial call to this
    ptrace::syscall(pid, None).unwrap();
    // There's no way this fails except if the child dies somehow
    let stat = wait::waitid(wait::Id::Pid(pid), WAIT_FLAGS).map_err(|_| ExecError::Died(None))?;
    match stat {
        // Again, report back death
        wait::WaitStatus::Exited(_, code) => {
            //eprintln!("Exited sig2 {code}");
            Err(ExecError::Died(Some(code)))
        }
        wait::WaitStatus::Signaled(_, _, _) => Err(ExecError::Died(None)),
        wait::WaitStatus::PtraceSyscall(pid) => {
            let regs = ptrace::getregs(pid).unwrap();
            if regs.syscall_nr() == syscall {
                Ok(regs)
            } else {
                panic!("Missed syscall while waiting for it to return: id {syscall}");
            }
        }
        // Should be impossible, but don't ever deadlock!
        _ => panic!("Somehow got stopped by signal while inside a syscall?"),
    }
}

/// Updates our state as needed following a page unmapping, removing that mapping
/// from our list if possible or registering it as an unmapping of other memory
/// otherwise.
///
/// TODO: Make this check if the page being unmapped belongs to the MiriMachine,
/// and determine if that's legal / how to handle it.
fn handle_munmap(
    pid: unistd::Pid,
    regs: libc::user_regs_struct,
    mmap_events: &mut Vec<MmapEvent>,
    page_size: usize,
) -> Result<(), ExecError> {
    // The unmap call might hit multiple mappings we've saved, so break it up
    // into individual pages
    let um_addr = regs.arg1();
    let um_count = regs.arg2().strict_div(page_size);

    // Indices of mappings we need to remove
    let mut idxes = vec![];
    // New unmappings that aren't just unmapping known state
    let mut to_append = vec![];

    // Iterate through mappings only and update the vecs above as needed
    for (idx, &mp) in mmap_events
        .iter()
        .filter_map(|mp| {
            match mp {
                MmapEvent::Mmap(addr) => Some(addr),
                MmapEvent::Munmap(_) => None,
            }
        })
        .enumerate()
    {
        for i in 0..um_count {
            // Either we're unmapping a page we know about, or this is some new
            // unmapping we should report back
            if mp == um_addr.strict_add(i.strict_mul(page_size)) {
                idxes.push(idx);
            } else {
                to_append.push(MmapEvent::Munmap(um_addr.strict_add(i.strict_mul(page_size))));
            }
        }
    }

    #[expect(clippy::as_conversions)]
    let regs = wait_for_syscall(pid, libc::SYS_munmap as _)?;

    // munmap returns 0 on success
    if regs.retval() == 0 {
        // We iterate thru this while removing elements so if
        // it's not reversed we will mess up the mappings badly!
        idxes.reverse();

        // Unmap succeeded, so take out the page(s) from our list and push the
        // new ones. No need to worry about partial unmaps because we only store
        // individual pages
        mmap_events.append(&mut to_append);
        for idx in idxes {
            mmap_events.remove(idx);
        }
    }

    Ok(())
}

/// Grabs the access that caused a segfault and logs it down if it's to our memory,
/// or kills the child and returns the appropriate error otherwise.
fn handle_segfault(
    pid: unistd::Pid,
    ch_pages: &[usize],
    ch_stack: usize,
    page_size: usize,
    cs: &capstone::Capstone,
    acc_events: &mut Vec<AccessEvent>,
) -> Result<(), ExecError> {
    /// This is just here to not pollute the main namespace with `capstone::prelude::*`.
    #[inline]
    fn capstone_disassemble(
        instr: &[u8],
        addr: usize,
        page_size: usize,
        cs: &capstone::Capstone,
        acc_events: &mut Vec<AccessEvent>,
    ) -> capstone::CsResult<()> {
        use capstone::prelude::*;

        // The arch_detail is what we care about, but it relies on these temporaries
        // that we can't drop. 0x1000 is the default base address for Captsone, and
        // we're expecting 1 instruction
        let insns = cs.disasm_count(instr, 0x1000, 1)?;
        let ins_detail = cs.insn_detail(&insns[0])?;
        let arch_detail = ins_detail.arch_detail();

        // Take an (addr, size, cutoff_size) and split an access into multiple if needed
        let get_ranges: fn(usize, usize, usize) -> Vec<std::ops::Range<usize>> =
            |addr, size, cutoff_size: usize| {
                let addr_added = addr.strict_add(size);
                let mut counter = 0usize;
                let mut ret = vec![];
                loop {
                    let curr = addr.strict_add(counter.strict_mul(cutoff_size));
                    let next = curr.strict_add(cutoff_size);
                    if next >= addr_added {
                        ret.push(curr..addr_added);
                        break;
                    } else {
                        ret.push(curr..curr.strict_add(cutoff_size));
                        counter = counter.strict_add(1);
                    }
                }
                ret
            };

        for op in arch_detail.operands() {
            match op {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                arch::ArchOperand::X86Operand(x86_operand) => {
                    match x86_operand.op_type {
                        // We only care about memory accesses
                        arch::x86::X86OperandType::Mem(_) => {
                            let append = get_ranges(addr, x86_operand.size.into(), page_size);
                            // It's called a "RegAccessType" but it also applies to memory
                            let acc_ty = x86_operand.access.unwrap();
                            if acc_ty.is_readable() {
                                acc_events.append(
                                    &mut append
                                        .clone()
                                        .into_iter()
                                        .map(AccessEvent::Read)
                                        .collect(),
                                );
                            }
                            if acc_ty.is_writable() {
                                acc_events.append(
                                    &mut append
                                        .clone()
                                        .into_iter()
                                        .map(AccessEvent::Write)
                                        .collect(),
                                );
                            }
                        }
                        _ => (),
                    }
                }
                #[cfg(target_arch = "aarch64")]
                arch::ArchOperand::Arm64Operand(arm64_operand) => {
                    // Annoyingly, we don't always get the size here, so just be pessimistic for now
                    match arm64_operand.op_type {
                        arch::arm64::Arm64OperandType::Mem(_) => {
                            // B = 1 byte, H = 2 bytes, S = 4 bytes, D = 8 bytes, Q = 16 bytes
                            let size = match arm64_operand.vas {
                                // Not an fp/simd instruction
                                arch::arm64::Arm64Vas::ARM64_VAS_INVALID => ARCH_WORD_SIZE,
                                // 1 byte
                                arch::arm64::Arm64Vas::ARM64_VAS_1B => 1,
                                // 2 bytes
                                arch::arm64::Arm64Vas::ARM64_VAS_1H => 2,
                                // 4 bytes
                                arch::arm64::Arm64Vas::ARM64_VAS_4B
                                | arch::arm64::Arm64Vas::ARM64_VAS_2H
                                | arch::arm64::Arm64Vas::ARM64_VAS_1S => 4,
                                // 8 bytes
                                arch::arm64::Arm64Vas::ARM64_VAS_8B
                                | arch::arm64::Arm64Vas::ARM64_VAS_4H
                                | arch::arm64::Arm64Vas::ARM64_VAS_2S
                                | arch::arm64::Arm64Vas::ARM64_VAS_1D => 8,
                                // 16 bytes
                                arch::arm64::Arm64Vas::ARM64_VAS_16B
                                | arch::arm64::Arm64Vas::ARM64_VAS_8H
                                | arch::arm64::Arm64Vas::ARM64_VAS_4S
                                | arch::arm64::Arm64Vas::ARM64_VAS_2D
                                | arch::arm64::Arm64Vas::ARM64_VAS_1Q => 16,
                            };
                            let append = get_ranges(addr, size, page_size);
                            // FIXME: This now has access type info in the latest
                            // git version of capstone because this pissed me off
                            // and I added it. Change this when it updates
                            acc_events.append(
                                &mut append.clone().into_iter().map(AccessEvent::Read).collect(),
                            );
                            acc_events.append(
                                &mut append.clone().into_iter().map(AccessEvent::Write).collect(),
                            );
                        }
                        _ => (),
                    }
                }
                #[cfg(target_arch = "arm")]
                arch::ArchOperand::ArmOperand(arm_operand) =>
                    match arm_operand.op_type {
                        arch::arm::ArmOperandType::Mem(_) => {
                            // We don't get info on the size of the access, but
                            // we're at least told if it's a vector inssizetruction
                            let size = if arm_operand.vector_index.is_some() {
                                ARCH_MAX_ACCESS_SIZE
                            } else {
                                ARCH_WORD_SIZE
                            };
                            let append = get_ranges(addr, size, page_size);
                            let acc_ty = arm_operand.access.unwrap();
                            if acc_ty.is_readable() {
                                acc_events.append(
                                    &mut append
                                        .clone()
                                        .into_iter()
                                        .map(AccessEvent::Read)
                                        .collect(),
                                );
                            }
                            if acc_ty.is_writable() {
                                acc_events.append(
                                    &mut append
                                        .clone()
                                        .into_iter()
                                        .map(AccessEvent::Write)
                                        .collect(),
                                );
                            }
                        }
                        _ => (),
                    },
                #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
                arch::ArchOperand::RiscVOperand(risc_voperand) => {
                    match risc_voperand {
                        arch::riscv::RiscVOperand::Mem(_) => {
                            // We get basically no info here
                            let append = get_ranges(addr, ARCH_MAX_ACCESS_SIZE, page_size);
                            acc_events.append(
                                &mut append.clone().into_iter().map(AccessEvent::Read).collect(),
                            );
                            acc_events.append(
                                &mut append.clone().into_iter().map(AccessEvent::Write).collect(),
                            );
                        }
                        _ => (),
                    }
                }
                _ => unimplemented!(),
            }
        }

        Ok(())
    }

    // Get information on what caused the segfault. This contains the address
    // that triggered it
    let siginfo = ptrace::getsiginfo(pid).map_err(|_| ExecError::Shrug)?;
    // All x86, ARM, etc. instructions only have at most one memory operand
    // (thankfully!)
    // SAFETY: si_addr is safe to call
    let addr = unsafe { siginfo.si_addr().addr() };
    let page_addr = addr.strict_sub(addr.strict_rem(page_size));

    if ch_pages.iter().any(|pg| (*pg..pg.strict_add(page_size)).contains(&addr)) {
        // Overall structure:
        // - Get the address that caused the segfault
        // - Unprotect the memory
        // - Step 1 instruction
        // - Parse executed code to estimate size & type of access
        // - Reprotect the memory
        // - Continue
        let stack_ptr = ch_stack.strict_add(FAKE_STACK_SIZE / 2);
        let regs_bak = ptrace::getregs(pid).unwrap();
        let mut new_regs = regs_bak;
        let ip_prestep = regs_bak.ip();

        // Move the instr ptr into the deprotection code
        #[expect(clippy::as_conversions)]
        new_regs.set_ip(mempr_off as usize);
        // Don't mess up the stack by accident!
        new_regs.set_sp(stack_ptr);

        // Modify the PAGE_ADDR global on the child process to point to the page
        // that we want unprotected
        ptrace::write(
            pid,
            (&raw const PAGE_ADDR).cast_mut().cast(),
            libc::c_long::try_from(page_addr).unwrap(),
        )
        .unwrap();

        // Check if we also own the next page, and if so unprotect it in case
        // the access spans the page boundary
        if ch_pages.contains(&page_addr.strict_add(page_size)) {
            ptrace::write(pid, (&raw const PAGE_COUNT).cast_mut().cast(), 2).unwrap();
        } else {
            ptrace::write(pid, (&raw const PAGE_COUNT).cast_mut().cast(), 1).unwrap();
        }

        ptrace::setregs(pid, new_regs).unwrap();

        // Our mempr_* functions end with a raise(SIGSTOP)
        wait_for_signal(pid, signal::SIGSTOP, true)?;

        // Step 1 instruction
        ptrace::setregs(pid, regs_bak).unwrap();
        ptrace::step(pid, None).unwrap();
        // Don't use wait_for_signal here since 1 instruction doesn't give room
        // for any uncertainty + we don't want it `cont()`ing randomly by accident
        // Also, don't let it continue with unprotected memory if something errors!
        let _ = wait::waitid(wait::Id::Pid(pid), WAIT_FLAGS).map_err(|_| ExecError::Died(None))?;

        // Save registers and grab the bytes that were executed. This would
        // be really nasty if it was a jump or similar but those thankfully
        // won't do memory accesses and so can't trigger this!
        let regs_bak = ptrace::getregs(pid).unwrap();
        new_regs = regs_bak;
        let ip_poststep = regs_bak.ip();
        // We need to do reads/writes in word-sized chunks
        let diff = (ip_poststep.strict_sub(ip_prestep)).div_ceil(ARCH_WORD_SIZE);
        let instr = (ip_prestep..ip_prestep.strict_add(diff)).fold(vec![], |mut ret, ip| {
            // This only needs to be a valid pointer in the child process, not ours
            ret.append(
                &mut ptrace::read(pid, std::ptr::without_provenance_mut(ip))
                    .unwrap()
                    .to_ne_bytes()
                    .to_vec(),
            );
            ret
        });

        // Now figure out the size + type of access and log it down
        // For now this will mark down e.g. the same area being read multiple
        // times, but that's still correct even if a bit inefficient
        if capstone_disassemble(&instr, addr, page_size, cs, acc_events).is_err() {
            // Read goes first because we need to be pessimistic
            acc_events.push(AccessEvent::Read(addr..addr.strict_add(ARCH_MAX_ACCESS_SIZE)));
            acc_events.push(AccessEvent::Write(addr..addr.strict_add(ARCH_MAX_ACCESS_SIZE)));
        }

        // Reprotect everything and continue
        #[expect(clippy::as_conversions)]
        new_regs.set_ip(mempr_on as usize);
        new_regs.set_sp(stack_ptr);
        ptrace::setregs(pid, new_regs).unwrap();
        wait_for_signal(pid, signal::SIGSTOP, true)?;

        ptrace::setregs(pid, regs_bak).unwrap();
        ptrace::syscall(pid, None).unwrap();
        Ok(())
    } else {
        // This was a real segfault, so print some debug info and quit
        let regs = ptrace::getregs(pid).unwrap();
        eprintln!("Segfault occurred during FFI at {addr:#018x}");
        eprintln!("Expected access on pages: {ch_pages:#018x?}");
        eprintln!("Register dump: {regs:#x?}");
        ptrace::kill(pid).unwrap();
        Err(ExecError::Died(None))
    }
}

/// Intercept the allocation/deallocation that happened upon calling `malloc`
/// or similar, logging them down. If the child dies, its return code is returned
/// as an error.
fn handle_sigtrap(
    pid: unistd::Pid,
    libc_events: &mut Vec<LibcEvent>,
    libc_vals: MagicLibcValues,
) -> Result<(), ExecError> {
    let regs = ptrace::getregs(pid).map_err(|_| ExecError::Shrug)?;
    // We'll be one instruction past the start
    match regs.ip().strict_sub(BREAKPT_INSTR_SIZE) {
        // malloc
        a if a == libc_vals.malloc_addr => {
            // Grab the size from registers and save it if the call is successful
            let size = regs.arg1();
            if let Ok(ptr) =
                intercept_retptr(pid, regs, libc_vals.malloc_addr, libc_vals.malloc_bytes)?
                    .try_into()
            {
                libc_events.push(LibcEvent::Malloc(ptr..ptr.strict_add(size)));
            }
        }
        // realloc
        a if a == libc_vals.realloc_addr => {
            // Free the old pointer, then mark down the new one
            let old_ptr = regs.arg1();
            let size = regs.arg2();
            // This can only match 1 item, unless malloc itself is misbehaving,
            // or it will error and this will be discarded
            let pos = libc_events.iter().position(|rg| {
                match rg {
                    // malloc will allow freeing pointers offset from their
                    // initial address as long as it's in the right range
                    LibcEvent::Malloc(rg) => rg.start <= old_ptr && old_ptr < rg.end,
                    LibcEvent::Free(_) => false,
                }
            });
            if let Ok(ptr) =
                intercept_retptr(pid, regs, libc_vals.realloc_addr, libc_vals.realloc_bytes)?
                    .try_into()
            {
                if let Some(pos) = pos {
                    // Freeing something we spotted during this run, so just pretend
                    // it never happened
                    libc_events.remove(pos);
                } else {
                    // Or it's removing a preexisting pointer, so we log this down
                    libc_events.push(LibcEvent::Free(old_ptr));
                }
                // Make sure it's ordered right! This goes at the end
                libc_events.push(LibcEvent::Malloc(ptr..ptr.strict_add(size)));
            }
        }
        // free
        a if a == libc_vals.free_addr => {
            let old_ptr = regs.arg1();
            let pos = libc_events.iter().position(|rg| {
                match rg {
                    LibcEvent::Malloc(rg) => rg.start <= old_ptr && old_ptr < rg.end,
                    LibcEvent::Free(_) => false,
                }
            });
            // This can lead to double-frees, but that's on the C code...
            // No real way for us to catch it here
            if let Some(pos) = pos {
                // Same as for realloc
                libc_events.remove(pos);
            } else {
                libc_events.push(LibcEvent::Free(old_ptr));
            }
            // Return value here doesn't exist, but make sure it doesn't error
            intercept_retptr(pid, regs, libc_vals.free_addr, libc_vals.free_bytes)?;
        }
        // This should almost definitely never happen, but better safe than sorry
        a => {
            eprintln!("Process got an unexpected SIGTRAP at addr {a:#018x?}; continuing...");
            ptrace::syscall(pid, None).unwrap();
        }
    }

    Ok(())
}

/// Gets the pointer or error value returned by the `libc` allocation functions
/// upon their being called. `fn_addr` should be the address of the respective
/// function, with `fn_bytes` being the original bytes it held before being
/// overwritten.
fn intercept_retptr(
    pid: unistd::Pid,
    mut regs: libc::user_regs_struct,
    fn_addr: usize,
    fn_bytes: isize,
) -> Result<isize, ExecError> {
    // Outline:
    // - Move instr ptr back before the sigtrap happened
    // - Restore the function to what it's supposed to be
    // - Change the function we're returning to so it gives us a sigtrap
    // - Catch it there
    // - Get the register-sized return value
    // - Patch the function back so it traps as before
    regs.set_ip(regs.ip().strict_sub(BREAKPT_INSTR_SIZE));
    // Just need to keep the same bit pattern
    #[expect(clippy::as_conversions)]
    let ret_addr = ptrace::read(pid, std::ptr::without_provenance_mut(regs.sp()))
        .map_err(|_| ExecError::Shrug)? as usize;
    let ret_bytes = ptrace::read(pid, std::ptr::without_provenance_mut(ret_addr)).unwrap();

    // Write a breakpoint at the return address
    // TODO: Make this more arch-agnostic
    ptrace::write(
        pid,
        std::ptr::without_provenance_mut(ret_addr),
        BREAKPT_INSTR.try_into().unwrap(),
    )
    .unwrap();
    // This one we did technically expose provenance for but it's in a different process anyways, so...
    #[expect(clippy::as_conversions)]
    ptrace::write(pid, std::ptr::without_provenance_mut(fn_addr), fn_bytes as _).unwrap();
    ptrace::setregs(pid, regs).unwrap();
    // Now wait for the function to return
    wait_for_signal(pid, signal::SIGTRAP, true)?;

    // We're getting the return value here
    let mut regs = ptrace::getregs(pid).unwrap();
    let ptr = regs.retval();
    regs.set_ip(regs.ip().strict_sub(BREAKPT_INSTR_SIZE));
    // Re-trap on allocation functions
    ptrace::write(
        pid,
        std::ptr::without_provenance_mut(fn_addr),
        BREAKPT_INSTR.try_into().unwrap(),
    )
    .unwrap();
    // And fix up the code we returned into
    ptrace::write(pid, std::ptr::without_provenance_mut(ret_addr), ret_bytes).unwrap();
    ptrace::setregs(pid, regs).unwrap();

    ptrace::syscall(pid, None).unwrap();
    Ok(ptr)
}

// We only get dropped into these functions via offsetting the instr pointer
// manually, so we *must not ever* unwind from them

/// Disables protections on the page whose address is currently in `PAGE_ADDR`.
///
/// SAFETY: `PAGE_ADDR` should be set to a page-aligned pointer to an owned page,
/// `PAGE_SIZE` should be the host pagesize, and the range from `PAGE_ADDR` to
/// `PAGE_SIZE` * `PAGE_COUNT` must be owned and allocated memory. No other threads
/// should be running.
pub unsafe extern "C" fn mempr_off() {
    use std::sync::atomic::Ordering;

    let len = PAGE_SIZE.load(Ordering::Relaxed).wrapping_mul(PAGE_COUNT.load(Ordering::Relaxed));
    // SAFETY: Upheld by caller
    unsafe {
        // It's up to the caller to make sure this doesn't actually overflow, but
        // we mustn't unwind from here, so...
        if libc::mprotect(
            PAGE_ADDR.load(Ordering::Relaxed).cast(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
        ) != 0
        {
            // Can't return or unwind, but we can do this
            std::process::exit(-1);
        }
    }
    // If this fails somehow we're doomed
    if signal::raise(signal::SIGSTOP).is_err() {
        std::process::exit(-1);
    }
}

/// Reenables protection on the page set by `PAGE_ADDR`.
///
/// SAFETY: See `mempr_off()`.
pub unsafe extern "C" fn mempr_on() {
    use std::sync::atomic::Ordering;

    let len = PAGE_SIZE.load(Ordering::Relaxed).wrapping_mul(PAGE_COUNT.load(Ordering::Relaxed));
    // SAFETY: Upheld by caller
    unsafe {
        if libc::mprotect(
            PAGE_ADDR.load(Ordering::Relaxed).cast(),
            len,
            libc::PROT_NONE,
        ) != 0
        {
            std::process::exit(-1);
        }
    }
    if signal::raise(signal::SIGSTOP).is_err() {
        std::process::exit(-1);
    }
}
