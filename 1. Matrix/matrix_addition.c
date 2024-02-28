#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Function to add two matrices
void add_Matrix(float *result, float *matrix_1, float *matrix_2, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i * cols + j] = matrix_1[i * cols + j] + matrix_2[i * cols + j];
        }
    }
}

int main() {
    int rows = 100;
    int cols = 100;

    // Allocate memory for matrices
    float *matrix_1 = (float *)malloc(rows * cols * sizeof(float));
    float *matrix_2 = (float *)malloc(rows * cols * sizeof(float));
    float *result = (float *)malloc(rows * cols * sizeof(float));

    // Populate matrices with random values (for testing)
    for (int i = 0; i < rows * cols; i++) {
        matrix_1[i] = (float)(rand() % 100);
        matrix_2[i] = (float)(rand() % 100);
    }

    // Perform matrix addition and measure execution time
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    add_Matrix(result, matrix_1, matrix_2, rows, cols);

    gettimeofday(&end, NULL);
    double execution_time = (end.tv_sec - start.tv_sec) * 1000.0; // Convert to milliseconds
    execution_time += (end.tv_usec - start.tv_usec) / 1000.0; // Convert to milliseconds

    printf("Execution time: %f ms\n", execution_time);

    // Free allocated memory
    free(matrix_1);
    free(matrix_2);
    free(result);

    return 0;
}
