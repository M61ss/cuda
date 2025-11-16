#include <cstdlib>
#include <iostream>

int main(void) {
    const int N = 1 << 20;

    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += rand() % 101;
    }
    double approx = sum / N;

    std::cout << "Montecarlo approximation is: " << approx << std::endl;
    
    return 0;
}