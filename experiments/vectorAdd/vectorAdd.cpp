#include <iostream>
#include <math.h>
#include <chrono>

void computeElapsedTime(std::chrono::_V2::system_clock::time_point begin, std::chrono::_V2::system_clock::time_point end, std::string msg)
{
    auto elapsed = (float)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000;
    std::cout << msg << ": " << elapsed << "ms" << std::endl;
}

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
    auto clock = std::chrono::high_resolution_clock();
    auto init = clock.now();

    auto begin = clock.now();
    float *v = (float *)malloc(N * sizeof(float));
    float *u = (float *)malloc(N * sizeof(float));
    float *label = (float *)malloc(N * sizeof(float));
    float *result = (float *)malloc(N * sizeof(float));
    auto end = clock.now();
    computeElapsedTime(begin, end, "Memory allocation time");

    begin = clock.now();
    for (int i = 0; i < N; i++)
    {
        v[i] = 1.0f;
        u[i] = 1.0f;
        label[i] = 2.0f;
    }
    end = clock.now();
    computeElapsedTime(begin, end, "Vector initialization time");

    begin = clock.now();
    vectorAdd(v, u, result, N);
    end = clock.now();
    computeElapsedTime(begin, clock.now(), "Vector sum time");

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

    end = clock.now();
    computeElapsedTime(init, end, "Total running time");

    exit(EXIT_SUCCESS);
}