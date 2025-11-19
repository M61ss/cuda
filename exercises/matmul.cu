#include <iostream>

void print_matrix(float **M, const int n, const int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            std::cout << M[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

__global__ void create_row(float *row)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    row[index] = 2.0f;
}

int main(void)
{
    const int N = 1 << 5;
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
    std::cout << "Memory allocation and prefetching terminated" << std::endl;

    int numT = 256;
    int numB = (N + numT - 1) / numT;
    for (int i = 0; i < N; i++) {
        create_row<<<numB, numT>>>(A[i]);
    }
    cudaDeviceSynchronize();
    std::cout << "Matrices creation terminated" << std::endl;

    print_matrix(A, N, N);
    // print_matrix(B, N, N);

    for (int i = 0; i < N; i++)
    {
        cudaFree(A[i]);
        cudaFree(B[i]);
    }
    cudaFree(A);
    cudaFree(B);
    return 0;
}