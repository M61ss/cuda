# Shared memory

## Avoid race condition

When sharing data between threads, we need to be careful to avoid **race conditions**, because while threads in a block run logically in parallel, not all threads can execute physically at the same time. Let’s say that two threads A and B each load a data element from global memory and store it to shared memory. Then, thread A wants to read B’s element from shared memory, and vice versa. Let’s assume that A and B are threads in two different warps. If B has not finished writing its element before A tries to read it, we have a race condition, which can lead to undefined behavior and incorrect results.
\
To ensure correct results when parallel threads cooperate, we must synchronize the threads.

It’s important to be aware that calling `__syncthreads()` in divergent code is undefined and can lead to deadlock—all threads within a thread block must call `__syncthreads()` at the same point.

## Static shared memory

The reason shared memory is used in the static shared memory example is to facilitate global memory coalescing on older CUDA devices (Compute Capability 1.1 or earlier). Optimal global memory coalescing is achieved for both reads and writes because global memory is always accessed through the linear, aligned index `t`. The reversed index `tr` is only used to access shared memory, which does not have the sequential access restrictions of global memory for optimal performance. The only performance issue with shared memory is bank conflicts.

## Dynamic shared memory

The dynamic shared memory kernel, `dynamicReverse()`, declares the shared memory array using an unsized extern array syntax, `extern __shared__ int s[]` (note the empty brackets and use of the extern specifier). The size is implicitly determined from the third execution configuration parameter when the kernel is launched. The remainder of the kernel code is identical to the `staticReverse()` kernel.

What if you need multiple dynamically sized arrays in a single kernel? You must declare a single extern unsized array as before, and use pointers into it to divide it into multiple arrays, as in the following excerpt:

```c++
extern __shared__ int s[];
int *integerData = s;                           // nI ints
float *floatData = (float*)&integerData[nI];    // nF floats
char *charData = (char*)&floatData[nF];         // nC chars
```

In the kernel launch, specify the total shared memory needed, as in the following:

```c++
myKernel<<<gridSize, blockSize, nI * sizeof(int) + nF * sizeof(float) + nC * sizeof(char)>>>(...);
```