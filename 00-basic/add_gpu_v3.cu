#include <iostream>
#include <math.h>

__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1 << 20;      // It shifts 1 to left 20 times => N = 2^20 = 1048576 (faster than using a function which computes the power)

    // Create two arrays of ~1M elements
    float *x, *y;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // Inizialize x and y on the device (GPU)
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on device
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before to accessing processed elements on host
    cudaDeviceSynchronize();

    // Check maximum error in calculations (all values of y should be 1.0 + 2.0 = 3.0)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}