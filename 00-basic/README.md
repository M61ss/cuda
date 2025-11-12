# Kernel function, threads, blocks

CUDA GPUs have many parallel processors grouped into Streaming Multiprocessors (SMs). Each SM can run multiple concurrent thread blocks, but each **thread block** runs on a single SM. At the same way, each thread block can run multiple **thread**, but each thread runs on a single block. 
\
Using more than one thread, computation is parallelized. Using more than one block, it is possible to further divide the work, enabling more parallelization. This way the program exploits all the available hardware. In other words, threads can be executed in parallel inside blocks which are run in parallel inside SM.

The purpose of the kernel function is to work on data in parallel. In order to do this, data have to be divided into subsets, each of which will be processed by a block executed in parallel with others in the SM. At the same way, these subsets have to be further separated in sub-subset, each of which will be processed by a thread executed in parallel with others in the block. 
\
For this reason, the kernel function has to know thread block dimension and thread index, so that a thread doesn't access memory locations on which others are working on.

```c++
__global__
void add(int n, float *x, float *y) {
    int index = threadIdx.x;    // Index of the current thread
    int stride = blockDim.x;    // Number of threads for every block
    
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}
```

The loop shown in the example code runs only on the data subset assigned to this thread and takes step proportional to the block dimension in order to not overlap its computation with other thread of the block (see image and formula below).

> [!IMPORTANT]
> 
> If I have N elements to process, and a certain amount of threads per block (blockSize), in order to get the maximum possible parallelization, I just need to calculate the number of blocks such as to get one thread per element, so at least N threads. 
> \
> This can be implemented simply dividing N by the block size (being careful to round up in case N is not a multiple of blockSize).
> ```c++
> int blockSize = 256;
> int numBlocks = (N + blockSize - 1) / blockSize;
> ```
>
> This choice in many cases is the best, but sometimes the scheduling task to assign every single element of vectors to a thread can generate a not negligible overhead. 

Together, the blocks of parallel threads make up what is known as the **grid**.

![grid](../resources/grid.png)

Assigning one element per thread, the index of the element processed by the thread is computed as:

$$
index=blockIdx.x*blockDim.x+threadaIdx.x
$$

It is also possible to specify the number of grids using the following syntax:

```c++
myKernel<<<gridNumber, blockNumber, threadNumber>>>(...)
```

Notice that `blockNumber` refers to number of blocks per grid, as well as `threadNumber` the number of threads per block.

# Unified Memory Prefetching

Unified Memory in CUDA is virtual memory. Individual virtual memory pages may be resident in the memory of any device (GPU or CPU) in the system, and those pages are migrated on demand. Since the memory pages are all CPU-resident when the kernel runs, there are multiple page faults and the hardware migrates the pages to the GPU memory when the faults occur. This results in a memory bottleneck, which is why we don’t see a speedup.
\
The migration is expensive because page faults occur individually, and GPU threads stall while they wait for the page migration. If we know what memory is needed by the kernel (x and y arrays), we can use prefetching to make sure that the data is on the GPU before the kernel needs it. We can do this by using the `cudaMemPrefetchAsync()` function before launching the kernel.