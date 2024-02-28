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

// Get OpenCL CUDA Information 
int opencl_cuda_info() {
    #define MAX_INFO_SIZE 1024 
    cl_platform_id platform;
    cl_device_id device;
    cl_uint num_platforms, num_devices;

    // Get the number of OpenCL platforms
    clGetPlatformIDs(1, &platform, &num_platforms);

    // Get the platform name
    char platform_name[MAX_INFO_SIZE];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, MAX_INFO_SIZE, platform_name, NULL);
    printf("OpenCL Platform: %s\n", platform_name);

    // Get the platform version
    char platform_version[MAX_INFO_SIZE];
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, MAX_INFO_SIZE, platform_version, NULL);
    printf("OpenCL Platform Version: %s\n", platform_version);

    // Get the number of OpenCL devices
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);

    // Get the device name
    char device_name[MAX_INFO_SIZE];
    clGetDeviceInfo(device, CL_DEVICE_NAME, MAX_INFO_SIZE, device_name, NULL);
    printf("OpenCL Device: %s\n", device_name);

    // Get the device version
    char device_version[MAX_INFO_SIZE];
    clGetDeviceInfo(device, CL_DEVICE_VERSION, MAX_INFO_SIZE, device_version, NULL);
    printf("OpenCL Device Version: %s\n", device_version);

    // Get the number of compute units
    cl_uint num_compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_compute_units, NULL);
    printf("OpenCL Device Compute Units: %u\n", num_compute_units);

    return 0;
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

    // Load kernel source code
    FILE *kernel_file = fopen("add_matrix.cl", "r");
    if (!kernel_file) {
        fprintf(stderr, "Failed to open kernel file.\n");
        exit(EXIT_FAILURE);
    }
    char *kernel_source;
    kernel_source = (char *)malloc(MAX_SOURCE_SIZE);
    size_t kernel_size = fread(kernel_source, 1, MAX_SOURCE_SIZE, kernel_file);
    fclose(kernel_file);

    // Initialize OpenCL
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint num_platforms, num_devices;
    cl_int err;

    // Get platform and device information
    err = clGetPlatformIDs(1, &platform_id, &num_platforms);
    checkError(err, "Failed to get platform ID");

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
    checkError(err, "Failed to get device ID");

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Failed to create context");

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    checkError(err, "Failed to create command queue");

    // Create memory buffers on the device for each matrix
    cl_mem matrix_1_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, rows * cols * sizeof(float), NULL, &err);
    checkError(err, "Failed to create buffer for matrix 1");

    cl_mem matrix_2_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, rows * cols * sizeof(float), NULL, &err);
    checkError(err, "Failed to create buffer for matrix 2");

    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, rows * cols * sizeof(float), NULL, &err);
    checkError(err, "Failed to create buffer for result");

    // Write matrices into the device memory
    err = clEnqueueWriteBuffer(command_queue, matrix_1_buffer, CL_TRUE, 0, rows * cols * sizeof(float), matrix_1, 0, NULL, NULL);
    checkError(err, "Failed to write matrix 1 to device");

    err = clEnqueueWriteBuffer(command_queue, matrix_2_buffer, CL_TRUE, 0, rows * cols * sizeof(float), matrix_2, 0, NULL, NULL);
    checkError(err, "Failed to write matrix 2 to device");

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, (const size_t *)&kernel_size, &err);
    checkError(err, "Failed to create program");

    // Build the program
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "Build error: %s\n", buffer);
        exit(1);
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "add_matrix", &err);
    checkError(err, "Failed to create kernel");

    // Set the arguments of the kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&matrix_1_buffer);
    checkError(err, "Failed to set kernel argument 0");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&matrix_2_buffer);
    checkError(err, "Failed to set kernel argument 1");

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&result_buffer);
    checkError(err, "Failed to set kernel argument 2");

    int num_elements = rows * cols;

    // Execute the OpenCL kernel
    size_t global_size[2] = {rows, cols};
    size_t local_size[2] = {10, 10}; // For example, you can adjust this as needed

    cl_event event;
    err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &event);
    checkError(err, "Failed to execute kernel");

    // Profiling
    clWaitForEvents(1, &event);
    cl_ulong start_time, end_time;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
    double execution_time = (end_time - start_time) * 1.0e-6; // Convert nanoseconds to milliseconds

    opencl_cuda_info();
    printf("Execution time on device: %f ms\n", execution_time);

    // Read the result buffer back to the host
    err = clEnqueueReadBuffer(command_queue, result_buffer, CL_TRUE, 0, num_elements * sizeof(float), result, 0, NULL, NULL);
    checkError(err, "Failed to read result buffer");

    // Free OpenCL resources
    clReleaseMemObject(matrix_1_buffer);
    clReleaseMemObject(matrix_2_buffer);
    clReleaseMemObject(result_buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    // Free allocated memory
    free(matrix_1);
    free(matrix_2);
    free(result);

    return 0;
}
