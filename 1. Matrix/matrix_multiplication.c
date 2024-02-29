#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 100 // Size of the matrix

// Function to multiply matrices
void multiply_Matrix(float *matrix1, float *matrix2, float *result) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            float sum = 0.0f;
            for (int k = 0; k < SIZE; k++) {
                sum += matrix1[i * SIZE + k] * matrix2[k * SIZE + j];
            }
            result[i * SIZE + j] = sum;
        }
    }
}

int main() {
    struct timeval start, end;
    float *matrix1, *matrix2, *result;

    // Allocate memory
    matrix1 = (float *)malloc(SIZE * SIZE * sizeof(float));
    matrix2 = (float *)malloc(SIZE * SIZE * sizeof(float));
    result = (float *)malloc(SIZE * SIZE * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < SIZE*SIZE; i++) {
        matrix1[i] = 1.0f; // Example value
        matrix2[i] = 2.0f; // Example value
    }

    // Profile the multiplication
    gettimeofday(&start, NULL);
    multiply_Matrix(matrix1, matrix2, result);
    gettimeofday(&end, NULL);

    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    float millis = micros / 1000.0; // Convert microseconds to milliseconds

    printf("Host Execution Time: %.2f ms\n", millis); // Print with 2 decimal places

    // Cleanup
    free(matrix1);
    free(matrix2);
    free(result);

    return 0;
}
