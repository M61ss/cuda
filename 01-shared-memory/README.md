# Shared memory

When sharing data between threads, we need to be careful to avoid **race conditions**, because while threads in a block run logically in parallel, not all threads can execute physically at the same time. Let’s say that two threads A and B each load a data element from global memory and store it to shared memory. Then, thread A wants to read B’s element from shared memory, and vice versa. Let’s assume that A and B are threads in two different warps. If B has not finished writing its element before A tries to read it, we have a race condition, which can lead to undefined behavior and incorrect results.
\
To ensure correct results when parallel threads cooperate, we must synchronize the threads.

It’s important to be aware that calling `__syncthreads()` in divergent code is undefined and can lead to deadlock—all threads within a thread block must call `__syncthreads()` at the same point.
