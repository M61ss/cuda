#include <iostream>
#include <math.h>
#include <chrono>

void computeElapsedTime(std::chrono::_V2::system_clock::time_point begin, std::chrono::_V2::system_clock::time_point end, std::string msg, float *total_time)
{
    auto elapsed = (float)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000;
    std::cout << msg << ": " << elapsed << "ms" << std::endl;
    *total_time += elapsed;
}

void vectorAdd(float *v, float *u, float *res, const int vector_length)
{
    for (int i = 0; i < vector_length; i++)
    {
        res[i] = v[i] + u[i];
    }
}

int main(void)
{
    const int vector_length = 1 << 20;
    // float v[vector_length], u[vector_length], y[vector_length], x[vector_length];    
    //                 ERROR: Setting vector_length too high causes stack overflow creating arrays
    //                        (the stack is 1MB), so I am using vectors
    const int vector_size = vector_length * sizeof(float);

    auto clock = std::chrono::high_resolution_clock();
    float total_time = 0.0f;

    auto begin = clock.now();
    float *v = (float *)malloc(vector_size);
    float *u = (float *)malloc(vector_size);
    float *label = (float *)malloc(vector_size);
    float *result = (float *)malloc(vector_size);
    auto end = clock.now();
    computeElapsedTime(begin, end, "Memory allocation time", &total_time);

    begin = clock.now();
    for (int i = 0; i < vector_length; i++)
    {
        v[i] = 1.0f;
        u[i] = 1.0f;
        label[i] = 2.0f;
    }
    end = clock.now();
    computeElapsedTime(begin, end, "Vector initialization time", &total_time);

    begin = clock.now();
    vectorAdd(v, u, result, vector_length);
    end = clock.now();
    computeElapsedTime(begin, clock.now(), "Vector sum time", &total_time);

    float maxError = 0;
    for (int i = 0; i < vector_length; i++)
    {
        maxError = fmaxf(0, fabsf(result[i] - label[i]));
    }
    std::cout << "Maximum detected error: " << maxError << std::endl;
    
    std::cout << "Total elapsed time: " << total_time << "ms" << std::endl;

    free(v);
    free(u);
    free(result);
    free(label);
    exit(EXIT_SUCCESS);
}