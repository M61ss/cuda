#include <iostream>
#include <chrono>

void computeElapsedTime(std::chrono::_V2::system_clock::time_point begin, std::chrono::_V2::system_clock::time_point end, std::string msg)
{
    auto elapsed = (float)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000;
    std::cout << msg << ": " << elapsed << "ms" << std::endl;
}

__global__ void vectorInit(float *v, float value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    v[i] = value;
}

__global__ void vectorAdd(float *v1, float *v2, float *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    result[i] = v1[i] + v2[i];
}

int main(void)
{
    const int N = 1 << 20;
    // auto clock = std::chrono::high_resolution_clock();       It causes warning "variable never used" because nvcc doesn't understand 
    //                                                          completly C++ type creation (nvcc has many limitations in C++ frontend,
    //                                                          so avoid fancy implementations)
    auto init = std::chrono::high_resolution_clock::now();

    auto begin = std::chrono::high_resolution_clock::now();
    float *v, *u, *label, *result;
    cudaMallocManaged(&v, N * sizeof(float));
    cudaMallocManaged(&u, N * sizeof(float));
    cudaMallocManaged(&label, N * sizeof(float));
    cudaMallocManaged(&result, N * sizeof(float));
    auto end = std::chrono::high_resolution_clock::now();
    computeElapsedTime(begin, end, "Unified Memory allocation time");

    cudaMemLocation loc;
    loc.id = 0;
    loc.type = cudaMemLocationTypeDevice;
    cudaMemPrefetchAsync(v, N * sizeof(float), loc, 0);        
    cudaMemPrefetchAsync(u, N * sizeof(float), loc, 0);     
    cudaMemPrefetchAsync(result, N * sizeof(float), loc, 0);
    int numThreads = 256;
    int numBlocks = (N + numThreads - 1) / numThreads;

    begin = std::chrono::high_resolution_clock::now();
    vectorInit<<<numBlocks, numThreads>>>(v, 1.0f);
    cudaDeviceSynchronize();
    vectorInit<<<numBlocks, numThreads>>>(u, 2.0f);
    cudaDeviceSynchronize();
    vectorInit<<<numBlocks, numThreads>>>(label, 3.0f);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    computeElapsedTime(begin, end, "Vector initialization time");

    begin = std::chrono::high_resolution_clock::now();
    vectorAdd<<<numBlocks, numThreads>>>(v, u, result);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    computeElapsedTime(begin, end, "Vector sum time");

    float maxError = 0;
    for (int i = 0; i < N; i++) {
        maxError = fmaxf(0, fabsf(result[i] - label[i]));
    }
    std::cout << "Maximum detected error: " << maxError << std::endl;

    cudaFree(v);
    cudaFree(u);
    cudaFree(result);
    cudaFree(label);

    end = std::chrono::high_resolution_clock::now();
    computeElapsedTime(init, end, "Total running time");

    exit(EXIT_SUCCESS);
}