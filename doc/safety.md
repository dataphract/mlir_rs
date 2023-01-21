# Modelling MLIR's safety rules

See the [MLIR Pass Management](https://mlir.llvm.org/docs/PassManagement/) documentation.

The MLIR pass manager supports running passes on multiple operations in parallel.
However, MLIR does not perform fine-grained synchronization internally, instead requiring that passes are well-behaved.
In particular, an instance of a pass runs on the contents of a single operation at a time, and:

- Must not modify the state of operations not nested within the operation being operated on.
- Must not modify the state of the operation being operated on in a manner visible to other passes.
- Must not inspect the internal state of sibling operations.
- Must not maintain mutable state between invocations on separate operations.
- Must not maintain any global mutable state.
- Must be clonable.

Even in single-threaded contexts, some operations in the C API invalidate existing handles.
API calls of the following forms have side effects visible to other handles:

- `mlirXDestroy`
- `mlirXYOwnedZ`
- `mlirXRemoveZ`
- `mlirXDetach`

These issues present a challenge when designing a safe Rust API on top of MLIR's C API.

## Thread safety

MLIR API calls broadly fall into one of two categories with regard to thread safety:

- Object creation and mutation calls which run through MLIR's `StorageUniquer` class are thread safe.
- All other API calls are not thread safe.

> As of this writing, in debug mode, MlirContext tracks an atomic flag `multiThreadedExecutionContext` to detect possible race conditions.
> This flag is distinct from the flag set by `mlirContextEnableMultithreading`:
> the former indicates that there is ongoing multithreaded work, while the latter indicates whether such work is allowed.

This makes representing `MlirContext` in Rust rather hairy:

- Almost every MLIR object holds a pointer to an associated `MlirContext` which must outlive it.
  This is natural to represent with a lifetime, like `Attribute<'cx>`.
- Creation of uniqued objects is synchronized.
- Creation of non-uniqued objects (dialects, for example) is not synchronized.
