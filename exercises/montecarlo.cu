#include <iostream>
#include <cstdlib>
#include <chrono>

__global__ void add(double *gpu_sum, double *samples)
{
    int index = gridDim.x * blockDim.x + threadIdx.x;
    *gpu_sum += samples[index];
}

int main(void)
{
    const int N = 1 << 20;
    double cpu_sum = 0;
    double gpu_sum = 0;
    double samples[N];
    for (int i = 0; i < N; i++)
    {
        samples[i] = rand() % 101;
    }
    int numThreads = 256;
    int numBlocks = (N + numThreads - 1) / numThreads;
    int numGrids = 1;

    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++)
    {
        cpu_sum += samples[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto et_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Elapsed time CPU: " << et_cpu.count() << "ms" << std::endl;

    begin = std::chrono::high_resolution_clock::now();
    add<<<numGrids, numBlocks, numThreads>>>(&gpu_sum, samples);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto et_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "Elapsed time CPU: " << et_gpu.count() << "ms" << std::endl;

    return 0;
}