#include <iostream>
#include <math.h>

int main(void)
{
    const int num_rows = 1 << 20;
    const int num_cols = 1 << 10;
    const int col_size = num_rows * sizeof(float *);
    const int row_size = num_cols * sizeof(float);
    const int label = 3.0f;

    float **A = (float **)malloc(col_size);
    float **B = (float **)malloc(col_size);
    float **C = (float **)malloc(col_size);

    for (int i = 0; i < num_rows; i++) {
        A[i] = (float *)malloc(row_size);
        B[i] = (float *)malloc(row_size);
        C[i] = (float *)malloc(row_size);
    }

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            A[i][j] = 1.0f;
            B[i][j] = 2.0f;
        }
    }

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            if (fabsf((C[i][j] - label) / label) > 2e-31) {
                fprintf(stderr, "Numerical error too large detected at (%d,%d)\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }

    for (int i = 0; i < num_rows; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    exit(EXIT_SUCCESS);
}