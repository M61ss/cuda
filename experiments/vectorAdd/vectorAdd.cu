__global__ void vectorInit(float *v, float value, const int length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    v[i] = value;
}

int main(void)
{
    const int N = 1 << 20;
    float *v, *u, *label, *result;
    cudaMallocManaged(&v, N * sizeof(float));
    cudaMallocManaged(&u, N * sizeof(float));
    cudaMallocManaged(&label, N * sizeof(float));
    cudaMallocManaged(&result, N * sizeof(float));

    cudaFree(v);
    cudaFree(u);
    cudaFree(result);
    cudaFree(label);
    exit(EXIT_SUCCESS);
}