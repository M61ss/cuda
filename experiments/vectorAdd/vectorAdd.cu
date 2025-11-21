#include <iostream>

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
    float *v, *u, *label, *result;
    cudaMallocManaged(&v, N * sizeof(float));
    cudaMallocManaged(&u, N * sizeof(float));
    cudaMallocManaged(&label, N * sizeof(float));
    cudaMallocManaged(&result, N * sizeof(float));

    cudaMemLocation loc;
    loc.id = 0;
    loc.type = cudaMemLocationTypeDevice;
    cudaMemPrefetchAsync(v, N * sizeof(float), loc, 0);        
    cudaMemPrefetchAsync(u, N * sizeof(float), loc, 0);     
    cudaMemPrefetchAsync(result, N * sizeof(float), loc, 0);
    int numThreads = 256;
    int numBlocks = (N + numThreads - 1) / numThreads;

    vectorInit<<<numBlocks, numThreads>>>(v, 1.0f);
    cudaDeviceSynchronize();
    vectorInit<<<numBlocks, numThreads>>>(u, 2.0f);
    cudaDeviceSynchronize();
    vectorInit<<<numBlocks, numThreads>>>(label, 3.0f);
    cudaDeviceSynchronize();

    vectorAdd<<<numBlocks, numThreads>>>(v, u, result);
    cudaDeviceSynchronize();

    float maxError = 0;
    for (int i = 0; i < N; i++) {
        maxError = fmaxf(0, fabsf(result[i] - label[i]));
    }
    std::cout << "Maximum detected error: " << maxError << std::endl;

    cudaFree(v);
    cudaFree(u);
    cudaFree(result);
    cudaFree(label);
    exit(EXIT_SUCCESS);
}