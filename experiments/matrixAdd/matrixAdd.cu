#include <iostream>

int main(void)
{
    const int num_rows = 1 << 10;
    const int num_cols = 1 << 5;

    const int col_size = num_rows * sizeof(float *);
    const int row_size = num_cols * sizeof(float);

    float **A, **B, **C;

    cudaMemLocation loc;
    loc.id = 0;
    loc.type = cudaMemLocationTypeDevice;

    cudaMallocManaged(&A, col_size);
    cudaMemPrefetchAsync(A, col_size, loc, 0);
    cudaMallocManaged(&B, col_size);
    cudaMemPrefetchAsync(B, col_size, loc, 0);
    cudaMallocManaged(&C, col_size);
    cudaMemPrefetchAsync(C, col_size, loc, 0);
    for (int i = 0; i < num_rows; i++)
    {
        cudaMallocManaged(&A[i], row_size);
        cudaMemPrefetchAsync(A[i], row_size, loc, 0);
        cudaMallocManaged(&B[i], row_size);
        cudaMemPrefetchAsync(B[i], row_size, loc, 0);
        cudaMallocManaged(&C[i], row_size);
        cudaMemPrefetchAsync(C[i], row_size, loc, 0);
    }



    for (int i = 0; i < num_rows; i++)
    {
        cudaFree(A[i]);
        cudaFree(B[i]);
        cudaFree(C[i]);
    }
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    exit(EXIT_SUCCESS);
}