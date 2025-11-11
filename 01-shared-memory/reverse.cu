#include <iostream>

__global__ void staticReverse(int *d, int n) {
    __shared__ int s[64];
    int t = threadIdx.x;
    int tr = n - t - 1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}

__global__ void dynamicReverse(int *d, int n) {
    extern __shared__ int s[];
    int t = threadIdx.x;
    int tr = n - t - 1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}

int main(void) {
    const int n = 64;
    int a[n], result[n], dst[n];

    for (int i = 0; i < n; i++) {
        a[i] = i;
        result[i] = n - i - 1;
        dst[i] = 0;
    }

    // Create a buffer of length n on the device as container where to copy host data  
    int *buffer_device;
    cudaMalloc(&buffer_device, n * sizeof(int));

    // Using static shared memory
    cudaMemcpy(buffer_device, a, n * sizeof(int), cudaMemcpyHostToDevice);
    staticReverse<<<1, n>>>(buffer_device, n);
    cudaMemcpy(dst, buffer_device, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        if (dst[i] != result[i]) {
            printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, dst[i], result[i]);
        }
    }

    // Using dynamic shared memory
    cudaMemcpy(buffer_device, a, n * sizeof(int), cudaMemcpyHostToDevice);
    dynamicReverse<<<1, n>>>(buffer_device, n);
    cudaMemcpy(dst, buffer_device, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        if (dst[i] != result[i]) {
            printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, dst[i], result[i]);
        }
    }

    return 0;
}