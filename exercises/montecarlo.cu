#include <iostream>
#include <cstdlib>
#include <chrono>

/*
    Demonstraits how shared memory bank conflicts lead GPU to have lower performance than CPU.
    Prefetch improves sightly performance. 
    Performance oscillates more on GPU than on CPU: 
        GPU (RTX 3060): 3ms~2ms 
        CPU (i7-10700k): 2.3ms~2.4ms
*/

__global__ void add(float *gpu_sum, float *samples)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    *gpu_sum += samples[index];
}

int main(void)
{
    const int N = 1 << 20;
    float cpu_sum = 0;
    float *samples;
    cudaMallocManaged(&samples, N * sizeof(float));
    for (int i = 0; i < N; i++)
    {
        samples[i] = rand() % 101;
    }

    int numThreads = 256;
    int numBlocks = (N + numThreads - 1) / numThreads;
    float *gpu_sum;
    cudaMallocManaged(&gpu_sum, sizeof(float));
    *gpu_sum = 0;
    cudaMemLocation loc;
    loc.id = 0;
    loc.type = cudaMemLocationTypeDevice;
    cudaMemPrefetchAsync(samples, N * sizeof(float), loc, 0);
    cudaMemPrefetchAsync(gpu_sum, sizeof(float), loc, 0);
    
    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
    {
        cpu_sum += samples[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto et_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "Result: " << cpu_sum / N << std::endl;
    std::cout << "Elapsed time CPU: " << (float)et_cpu.count() / 1000 << "ms" << std::endl;

    begin = std::chrono::high_resolution_clock::now();
    add<<<numBlocks, numThreads>>>(gpu_sum, samples);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto et_gpu = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cout << "Result: " << *gpu_sum / N << std::endl;
    std::cout << "Elapsed time GPU: " << (float)et_gpu.count() / 1000 << "ms" << std::endl;

    cudaFree(samples);
    return 0;
}