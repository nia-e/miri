mod child;
mod parent;

use std::ops::Range;

pub use self::child::{Supervisor, init_sv, register_retcode_sv};

/// The size used for the array into which we can move the stack pointer.
const FAKE_STACK_SIZE: usize = 1024;

/// Information needed to begin tracing.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct StartFfiInfo {
    /// A vector of page addresses. These should have been automatically obtained
    /// with `IsolatedAlloc::pages` and prepared with `IsolatedAlloc::prepare_ffi`.
    page_ptrs: Vec<usize>,
    /// The address of an allocation that can serve as a temporary stack.
    /// This should be a leaked `Box<[u8; FAKE_STACK_SIZE]>` cast to an int.
    stack_ptr: usize,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
enum TraceRequest {
    StartFfi(StartFfiInfo),
    OverrideRetcode(i32),
}

/// A single memory access, conservatively overestimated
/// in case of ambiguity.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum AccessEvent {
    /// A read may have occurred on no more than the specified address range.
    Read(Range<usize>),
    /// A write may have occurred on no more than the specified address range.
    Write(Range<usize>),
}

/// The result(s) of a call to a `libc` allocation-related function. Note that
/// some function e.g. `realloc` will generate multiple events (an allocation
/// and a deallocation).
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum LibcEvent {
    /// A heap allocation was created spanning the addresses in the range.
    Malloc(Range<usize>),
    /// A pointer with the inner address was `free`d.
    Free(usize),
}

/// A singular page mapping or unmapping. The inner field is always a page-aligned
/// address, representing a single system page being mapped/unmapped.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum MmapEvent {
    /// The range from this address to one system page past it was mapped to
    /// memory.
    Mmap(usize),
    /// The range from this address to one system page past it was unmapped
    /// from memory
    Munmap(usize),
}

/// The final results of an FFI trace, containing every relevant event detected
/// by the tracer.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct MemEvents {
    /// An ordered list of memory accesses that occurred.
    pub acc_events: Vec<AccessEvent>,
    /// A value modulo which `AccessEvent` ranges stay the same length. Makes
    /// parsing the events a lot easier. Should likely just be the page size.
    pub alloc_cutoff: usize,
    /// An ordered list of libc events that occurred. `malloc`s which were `free`d
    /// before the end of the FFI call will have been removed on a best-effort
    /// basis, but nothing else.
    pub libc_events: Vec<LibcEvent>,
    /// An ordered list of page mappings and unmappings that occurred. Same as
    /// for `libc_events`, mappings that were unmapped will have been removed,
    /// but nothing else.
    pub mmap_events: Vec<MmapEvent>,
}
