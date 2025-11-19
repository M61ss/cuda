#include <iostream>

__global__ void create_row(float *row)
{
}

__global__ void create(float **A)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
}

int main(void)
{
    const int N = 1 << 20;
    float **A, **B;
    cudaMallocManaged(&A, N * sizeof(float *));
    cudaMallocManaged(&B, N * sizeof(float *));
    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeDevice;
    loc.id = 0;
    for (int i = 0; i < N; i++)
    {
        cudaMallocManaged(&A[i], N * sizeof(float));
        cudaMemPrefetchAsync(A[i], N * sizeof(float), loc, 0);
        cudaMallocManaged(&B[i], N * sizeof(float));
        cudaMemPrefetchAsync(B[i], N * sizeof(float), loc, 0);
    }
    cudaMemPrefetchAsync(A, N * sizeof(float *), loc, 0);
    cudaMemPrefetchAsync(B, N * sizeof(float *), loc, 0);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            
        }
    }

    for (int i = 0; i < N; i++)
    {
        cudaFree(A[i]);
        cudaFree(B[i]);
    }
    cudaFree(A);
    cudaFree(B);
    return 0;
}