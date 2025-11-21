#include <iostream>
#include <math.h>
#include <chrono>

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

    auto elapsed = (float)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000;
    std::cout << "Memory allocation took: " << elapsed << "ms" << std::endl;

    begin = clock.now();
    for (int i = 0; i < N; i++)
    {
        v[i] = 1.0f;
        u[i] = 1.0f;
        label[i] = 2.0f;
    }
    end = clock.now();

    elapsed = (float)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000;
    std::cout << "Vector initialization took: " << elapsed << "ms" << std::endl;

    begin = clock.now();
    vectorAdd(v, u, result, N);
    end = clock.now();

    elapsed = (float)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000;
    std::cout << "Vector add took: " << elapsed << "ms" << std::endl;

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

    auto total_exec_time = (float)std::chrono::duration_cast<std::chrono::microseconds>(clock.now() - init).count() / 1000;
    std::cout << "Total running time: " << total_exec_time  << "ms" << std::endl;

    exit(EXIT_SUCCESS);
}