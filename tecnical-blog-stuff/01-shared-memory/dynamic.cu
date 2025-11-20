#include <iostream>

__global__ void dynamicReverse(int *d, int n) {
    // Declaring an array using dynamic shared memory
    extern __shared__ int s[];
    int t = threadIdx.x;
    int t_reverse = n - t - 1;
    s[t] = d[t];
    // We need to sync threads because before to use data stored in s, the whole array has to be filled
    __syncthreads();
    d[t] = s[t_reverse];
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

    // Using dynamic shared memory
    cudaMemcpy(buffer_device, a, n * sizeof(int), cudaMemcpyHostToDevice);
    dynamicReverse<<<1, n, n * sizeof(int)>>>(buffer_device, n);
    cudaMemcpy(dst, buffer_device, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        if (dst[i] != result[i]) {
            printf("Error: d[%d]!=r[%d] (%d, %d)n", i, i, dst[i], result[i]);
        }
    }

    return 0;
}