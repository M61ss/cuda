# Shared memory

## Avoid race condition

When sharing data between threads, we need to be careful to avoid **race conditions**, because while threads in a block run logically in parallel, not all threads can execute physically at the same time. Let’s say that two threads A and B each load a data element from global memory and store it to shared memory. Then, thread A wants to read B’s element from shared memory, and vice versa. Let’s assume that A and B are threads in two different warps. If B has not finished writing its element before A tries to read it, we have a race condition, which can lead to undefined behavior and incorrect results.
\
To ensure correct results when parallel threads cooperate, we must synchronize the threads.

It’s important to be aware that calling `__syncthreads()` in divergent code is undefined and can lead to deadlock—all threads within a thread block must call `__syncthreads()` at the same point.

## Static shared memory

The reason shared memory is used in the static shared memory example is to facilitate global memory coalescing on older CUDA devices (Compute Capability 1.1 or earlier). Optimal global memory coalescing is achieved for both reads and writes because global memory is always accessed through the linear, aligned index `t`. The reversed index `tr` is only used to access shared memory, which does not have the sequential access restrictions of global memory for optimal performance. The only performance issue with shared memory is bank conflicts.

## Dynamic shared memory

