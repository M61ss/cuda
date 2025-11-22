#include <iostream>
#include <chrono>

void computeElapsedTime(std::chrono::_V2::system_clock::time_point begin, std::chrono::_V2::system_clock::time_point end, std::string msg, float *total_time)
{
    auto elapsed = (float)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000;
    std::cout << msg << ": " << elapsed << "ms" << std::endl;

    *total_time += elapsed;
}

__global__ void vectorInit(float *v, float value, const int vector_length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // This check is necessary to avoid segmentation fault because not all vector are power of 2 long.
    // In general, often the number of threads doesn't correspond to the vector length 
    if (i < vector_length)
        v[i] = value;
}

__global__ void vectorAdd(float *v1, float *v2, float *result, const int vector_length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < vector_length)
        result[i] = v1[i] + v2[i];
}

int main(void)
{
    const int vector_length = 1 << 20;
    
    cudaError_t error = cudaSuccess;
    
    float total_time = 0.0f;

    // MEMORY ALLOCATION

    // auto clock = std::chrono::high_resolution_clock();       It causes warning "variable never used" because nvcc doesn't understand 
    //                                                          completly C++ type creation (nvcc has many limitations in C++ frontend,
    //                                                          so avoid fancy implementations)
    auto begin = std::chrono::high_resolution_clock::now();

    float *v, *u, *label, *result;

    cudaMallocManaged(&v, vector_length * sizeof(float));

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate vector 'v' on Unified Memory. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaMallocManaged(&u, vector_length * sizeof(float));

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate vector 'u' on Unified Memory. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaMallocManaged(&label, vector_length * sizeof(float));

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate vector 'label' on Unified Memory. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaMallocManaged(&result, vector_length * sizeof(float));

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate vector 'result' on Unified Memory. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    auto end = std::chrono::high_resolution_clock::now();
    computeElapsedTime(begin, end, "Unified Memory allocation time", &total_time);

    // MEMORY PREFETCH

    cudaMemLocation loc;
    loc.id = 0;
    loc.type = cudaMemLocationTypeDevice;

    cudaMemPrefetchAsync(v, vector_length * sizeof(float), loc, 0);

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to prefetch vector 'v' on device. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }        

    cudaMemPrefetchAsync(u, vector_length * sizeof(float), loc, 0);

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to prefetch vector 'u' on device. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    cudaMemPrefetchAsync(label, vector_length * sizeof(float), loc, 0);

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to prefetch vector 'label' on device. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    } 

    cudaMemPrefetchAsync(result, vector_length * sizeof(float), loc, 0);

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to prefetch vector 'result' on device. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }  

    // VECTOR INITIALIZATION

    int numThreads = 256;
    int numBlocks = (vector_length + numThreads - 1) / numThreads;

    begin = std::chrono::high_resolution_clock::now();

    vectorInit<<<numBlocks, numThreads>>>(v, 1.0f, vector_length);

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to initialize vector 'v' on device. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    vectorInit<<<numBlocks, numThreads>>>(u, 2.0f, vector_length);

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to initialize vector 'u' on device. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    vectorInit<<<numBlocks, numThreads>>>(label, 3.0f, vector_length);

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to initialize vector 'label' on device. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to sync device with host. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    end = std::chrono::high_resolution_clock::now();
    computeElapsedTime(begin, end, "Vector initialization time", &total_time);

    // VECTOR SUM

    begin = std::chrono::high_resolution_clock::now();

    vectorAdd<<<numBlocks, numThreads>>>(v, u, result, vector_length);

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to compute v+u on device. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to sync device with host. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    end = std::chrono::high_resolution_clock::now();
    computeElapsedTime(begin, end, "Vector sum time", &total_time);

    // COMPUTE NUMERICAL ERROR

    float maxError = 0;

    for (int i = 0; i < vector_length; i++) {
        maxError = fmaxf(0, fabsf(result[i] - label[i]));
    }
    std::cout << "Maximum detected error: " << maxError << std::endl;

    std::cout << "Total elapsed time: " << total_time << "ms" << std::endl;

    // FREE MEMORY

    cudaFree(v);

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to free 'v' memory. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaFree(u);

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to free 'u' memory. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaFree(result);

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to free 'result' memory. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaFree(label);

    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to free 'label' memory. Code %s", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    exit(EXIT_SUCCESS);
}