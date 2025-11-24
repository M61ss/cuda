#include <iostream>

__global__ void matrixAdd(float **A, float **B, float **C, const int num_rows, const int num_cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < num_rows && j < num_cols)
        C[i][j] = A[i][j] + B[i][j];
}

int main(void)
{
    const int num_rows = 1 << 10;
    const int num_cols = 1 << 5;

    const int col_size = num_rows * sizeof(float *);
    const int row_size = num_cols * sizeof(float);

    float **A, **B, **C;

    dim3 numThreads(16, 16);
    dim3 numBlocks(num_rows / numThreads.x, num_cols / numThreads.y);

    cudaError_t error = cudaSuccess;

    // MEMORY ALLOCATION AND PREFETCH

    cudaMemLocation loc;
    loc.id = 0;
    loc.type = cudaMemLocationTypeDevice;

    cudaMallocManaged(&A, col_size);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate Unified Memory for matrix 'A' rows. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaMemPrefetchAsync(A, col_size, loc, 0);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to prefetch to device matrix 'A' rows. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaMallocManaged(&B, col_size);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate Unified Memory for matrix 'B' rows. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaMemPrefetchAsync(B, col_size, loc, 0);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to prefetch to device matrix 'B' rows. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaMallocManaged(&C, col_size);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate Unified Memory for matrix 'C' rows. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaMemPrefetchAsync(C, col_size, loc, 0);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to prefetch to device matrix 'C' rows. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_rows; i++)
    {
        cudaMallocManaged(&A[i], row_size);

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate Unified Memory for matrix 'A' cols for row '%d'. Code %s", i, cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        cudaMemPrefetchAsync(A[i], row_size, loc, 0);

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to prefetch to device matrix 'A' cols for row '%d'. Code %s", i, cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        cudaMallocManaged(&B[i], row_size);

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate Unified Memory for matrix 'B' cols for row '%d'. Code %s", i, cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        cudaMemPrefetchAsync(B[i], row_size, loc, 0);

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to prefetch to device matrix 'B' cols for row '%d'. Code %s", i, cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        cudaMallocManaged(&C[i], row_size);

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate Unified Memory for matrix 'C' cols for row '%d'. Code %s", i, cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        cudaMemPrefetchAsync(C[i], row_size, loc, 0);

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to prefetch to device matrix 'C' cols for row '%d'. Code %s", i, cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    }

    // MATRIX INITIALIZATION

    

    // MATRIX SUM

    matrixAdd<<<numBlocks, numThreads>>>(A, B, C, num_rows, num_cols);

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to perform C=A+B. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // FREE MEMROY

    for (int i = 0; i < num_rows; i++)
    {
        cudaFree(A[i]);

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to free matrix 'A' cols for row '%d'. Code %s", i, cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        cudaFree(B[i]);

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to free matrix 'B' cols for row '%d'. Code %s", i, cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        cudaFree(C[i]);

        if (error != cudaSuccess)
        {
            fprintf(stderr, "Failed to free matrix 'C' cols for row '%d'. Code %s", i, cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }
    }

    cudaFree(A);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free matrix 'A' rows. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaFree(B);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free matrix 'B' rows. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaFree(C);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to free matrix 'C' rows. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}