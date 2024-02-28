#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

void checkError(cl_int error, const char *message) {
    if (error != CL_SUCCESS) {
        printf("%s: %d\n", message, error);
        exit(EXIT_FAILURE);
    }
}

int opencl_cuda_info() {
    // Your function implementation remains the same
}

int main() {
    int rowsA = 100, colsA = 100;
    int rowsB = 100, colsB = 100;
    int rowsC = rowsA, colsC = colsB; // For matrix multiplication, C is MxK

    if (colsA != rowsB) {
        printf("Error: Matrix dimensions must match for multiplication!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for matrices
    float *matrix_A = (float *)malloc(rowsA * colsA * sizeof(float));
    float *matrix_B = (float *)malloc(rowsB * colsB * sizeof(float));
    float *matrix_C = (float *)malloc(rowsC * colsC * sizeof(float));

    // Populate matrices A and B with random values
    for (int i = 0; i < rowsA * colsA; i++) {
        matrix_A[i] = (float)(rand() % 100);
    }
    for (int i = 0; i < rowsB * colsB; i++) {
        matrix_B[i] = (float)(rand() % 100);
    }

    // Load the kernel source code
    FILE *file = fopen("multiply_matrix.cl", "r");
    if (!file) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(EXIT_FAILURE);
    }
    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, file);
    fclose(file);

    // Initialize OpenCL
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    checkError(ret, "Failed to get platform IDs");

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    checkError(ret, "Failed to get device IDs");

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    checkError(ret, "Failed to create context");

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
    checkError(ret, "Failed to create command queue");

    // Create memory buffers
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, rowsA * colsA * sizeof(float), NULL, &ret);
    checkError(ret, "Failed to create buffer for A");

    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, rowsB * colsB * sizeof(float), NULL, &ret);
    checkError(ret, "Failed to create buffer for B");

    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, rowsC * colsC * sizeof(float), NULL, &ret);
    checkError(ret, "Failed to create buffer for C");

    // Copy the matrices to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, rowsA * colsA * sizeof(float), matrix_A, 0, NULL, NULL);
    checkError(ret, "Failed to write to buffer A");

    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, rowsB * colsB * sizeof(float), matrix_B, 0, NULL, NULL);
    checkError(ret, "Failed to write to buffer B");

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    checkError(ret, "Failed to create program");

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        // Determine the reason for the error
        char build_log[16384];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
        printf("Error in kernel build:\n%s\n", build_log);
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "multiply_matrix", &ret);
    checkError(ret, "Failed to create kernel");

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(int), &rowsA);
    checkError(ret, "Failed to set argument 0");
    ret = clSetKernelArg(kernel, 1, sizeof(int), &colsA);
    checkError(ret, "Failed to set argument 1");
    ret = clSetKernelArg(kernel, 2, sizeof(int), &colsB);
    checkError(ret, "Failed to set argument 2");
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&a_mem_obj);
    checkError(ret, "Failed to set argument 3");
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&b_mem_obj);
    checkError(ret, "Failed to set argument 4");
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&c_mem_obj);
    checkError(ret, "Failed to set argument 5");

    // Execute the OpenCL kernel on the matrix
    size_t global_item_size[2] = {rowsC, colsC}; // Process the entire matrix
    size_t local_item_size[2] = {16, 16}; // Divide work items into groups of 16
    cl_event kernel_event;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, &kernel_event);
    checkError(ret, "Failed to enqueue NDRange kernel");

    // Wait for the kernel to complete
    ret = clWaitForEvents(1, &kernel_event);
    checkError(ret, "Failed to wait for kernel event");

    // Profiling the execution time
    cl_ulong time_start, time_end;
    ret = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    checkError(ret, "Failed to get profiling info (start)");

    ret = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    checkError(ret, "Failed to get profiling info (end)");

    double execution_time_ms = (time_end - time_start) / 1000000.0; // Convert from nanoseconds to milliseconds
    printf("Kernel execution time: %f ms\n", execution_time_ms);

    // Copy the result from the device to the host
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, rowsC * colsC * sizeof(float), matrix_C, 0, NULL, NULL);
    checkError(ret, "Failed to read output array C");

    // Display a small part of the result for verification
    printf("Result matrix:\n");
    for (int i = 0; i < 10; ++i) { // Just print part of the matrix for brevity
        printf("%f ", matrix_C[i]);
        if ((i + 1) % colsC == 0) printf("\n");
    }

    // Clean up
    ret = clReleaseEvent(kernel_event);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(matrix_A);
    free(matrix_B);
    free(matrix_C);
    free(source_str);

    // Optionally, you can call your opencl_cuda_info function here
    // opencl_cuda_info();

    return 0;
}

