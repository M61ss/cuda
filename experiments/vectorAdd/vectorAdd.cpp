#include <iostream>
#include <math.h>

void vectorAdd(float *v, float *u, float *res, const int N)
{
    for (int i = 0; i < N; i++)
    {
        res[i] = v[i] + u[i];
    }
}

int main(void)
{
    const int N = 1 << 20;
    // float v[N], u[N], y[N], x[N];    ERROR: Setting N too high causes stack overflow creating arrays
    //                                         (the stack is 1MB), so I am using vectors
    float *v = (float *)malloc(N * sizeof(float));
    float *u = (float *)malloc(N * sizeof(float));
    float *label = (float *)malloc(N * sizeof(float));
    float *result = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        v[i] = 1.0f;
        u[i] = 1.0f;
        label[i] = 2.0f;
    }

    vectorAdd(v, u, result, N);

    float maxError = 0;
    for (int i = 0; i < N; i++)
    {
        maxError = fmaxf(0, fabsf(result[i] - label[i]));
    }
    std::cout << "Maximum detected error: " << maxError << std::endl;

    free(v);
    free(u);
    free(result);
    free(label);
    exit(EXIT_SUCCESS);
}