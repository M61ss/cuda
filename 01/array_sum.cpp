#include <iostream>
#include <math.h>

void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1 << 20;      // It shifts 1 to left 20 times => N = 2^20 = 1048576 (faster than using a function which computes the power)

    // Create two arrays of ~1M elements
    float *x = new float[N];
    float *y = new float[N];

    // Inizialize x and y on the host (CPU)
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on host
    add(N, x, y);

    // Check maximum error in calculations (all values of y should be 1.0 + 2.0 = 3.0)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete [] x;
    delete [] y;

    return 0;
}