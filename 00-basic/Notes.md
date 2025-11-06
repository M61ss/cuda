# Kernel function, threads and blocks

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

The loop shown in the example code runs only on the data subset assigned to this thread and takes step proportional to the block dimension.

Together, the blocks of parallel threads make up what is known as the **grid**.