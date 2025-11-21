__global__ void vectorInit(float *v, float value, const int length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    v[i] = value;
}

int main(void)
{
    const int N = 1 << 20;
    float *v = (float *)malloc(N * sizeof(float));
    float *u = (float *)malloc(N * sizeof(float));
    float *result = (float *)malloc(N * sizeof(float));
    float *label = (float *)malloc(N * sizeof(float));

    free(v);
    free(u);
    free(result);
    free(label);
    exit(EXIT_SUCCESS);
}