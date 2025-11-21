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
    float *y = (float *)malloc(N * sizeof(float));
    float *x = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        v[i] = 1.0f;
        u[i] = 1.0f;
        y[i] = 2.0f;
    }

    vectorAdd(v, u, x, N);

    float maxError = 0;
    for (int i = 0; i < N; i++)
    {
        maxError = fmaxf(0, fabsf(x[i] - y[i]));
    }
    std::cout << "Maximum detected error: " << maxError << std::endl;

    free(v);
    free(u);
    free(x);
    free(y);

    exit(EXIT_SUCCESS);
}