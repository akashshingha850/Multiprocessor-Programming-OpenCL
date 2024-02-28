#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

void checkError(cl_int error, const char *message) {
    if (error != CL_SUCCESS) {
        fprintf(stderr, "%s: %d\n", message, error);
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Initialize matrices dimensions
    const int M = 100, N = 100, K = 100;

    // Allocate memory for matrices A, B, and C
    float *A = (float *)malloc(M * N * sizeof(float));
    float *B = (float *)malloc(N * K * sizeof(float));
    float *C = (float *)malloc(M * K * sizeof(float));

    // Populate matrices A and B with random values for demonstration
    for (int i = 0; i < M * N; i++) {
        A[i] = rand() % 100;
    }
    for (int i = 0; i < N * K; i++) {
        B[i] = rand() % 100;
    }

    // Load kernel source code
    FILE *file = fopen("multiply_matrix.cl", "r");
    if (!file) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(EXIT_FAILURE);
    }
    char *source_str = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, file);
    fclose(file);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    checkError(ret, "Failed to get platform IDs");

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    checkError(ret, "Failed to get device IDs");

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create command queue with profiling enabled
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    // Create memory buffers on the device
    cl_mem memobjA = clCreateBuffer(context, CL_MEM_READ_ONLY, M * N * sizeof(float), NULL, &ret);
    cl_mem memobjB = clCreateBuffer(context, CL_MEM_READ_ONLY, N * K * sizeof(float), NULL, &ret);
    cl_mem memobjC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, M * K * sizeof(float), NULL, &ret);

    // Copy matrices to the device
    ret = clEnqueueWriteBuffer(command_queue, memobjA, CL_TRUE, 0, M * N * sizeof(float), A, 0, NULL, NULL);
    ret |= clEnqueueWriteBuffer(command_queue, memobjB, CL_TRUE, 0, N * K * sizeof(float), B, 0, NULL, NULL);
    checkError(ret, "Failed to write data to device");

    // Create program from kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        // Determine the reason for the error
        char build_log[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
        fprintf(stderr, "Error in kernel build:\n%s\n", build_log);
        exit(1);
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "multiply_matrix", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(int), (void *)&M);
    ret |= clSetKernelArg(kernel, 1, sizeof(int), (void *)&N);
    ret |= clSetKernelArg(kernel, 2, sizeof(int), (void *)&K);
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&memobjA);
    ret |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&memobjB);
    ret |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&memobjC);
    checkError(ret, "Failed to set kernel arguments");

    // Execute the OpenCL kernel
    size_t global_item_size[2] = {M, K}; // Process the entire lists
    cl_event event;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, NULL, 0, NULL, &event); // NULL local work size lets OpenCL decide
    checkError(ret, "Failed to enqueue NDRange kernel");

    // Wait for the kernel to complete
    ret = clFinish(command_queue);
    checkError(ret, "Waiting for kernel to finish");

    // Profiling: Get execution time
    cl_ulong time_start, time_end;
    ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    ret |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    checkError(ret, "Failed to get event profiling info");

    double execution_time_ms = (double)(time_end - time_start) * 1e-6; // Convert from nanoseconds to milliseconds
    printf("Kernel Execution Time: %0.3f ms\n", execution_time_ms);

    // Read the result back to the host
    ret = clEnqueueReadBuffer(command_queue, memobjC, CL_TRUE, 0, M * K * sizeof(float), C, 0, NULL, NULL);
    checkError(ret, "Failed to read output array C");

    // Cleanup
    ret = clReleaseKernel(kernel);
    ret |= clReleaseProgram(program);
    ret |= clReleaseMemObject(memobjA);
    ret |= clReleaseMemObject(memobjB);
    ret |= clReleaseMemObject(memobjC);
    ret |= clReleaseCommandQueue(command_queue);
    ret |= clReleaseContext(context);
    checkError(ret, "Failed during cleanup");

    free(A);
    free(B);
    free(C);
    free(source_str);

    return 0;
}
