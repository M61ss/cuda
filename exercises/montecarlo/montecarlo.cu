#include <iostream>
#include <cstdlib>

__global__ void sample(double *sum, double *data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    *sum += data[index];
}

int main(void) {
    const int N = 1 << 20;
    double data[N];
    for (int i = 0; i < N; i++) {
        data[i] = rand() % 101;
    }
    double sum = 0;

    int numThreads = 256;
    int numBlocks = (N + numThreads - 1) / numThreads;
    int numGrids = 1;
    sample<<<numGrids, numBlocks, numThreads>>>(&sum, data);
    cudaDeviceSynchronize();

    double approx = sum / N;
    std::cout << "Montecarlo approximation is: " << approx << std::endl;

    return 0;
}