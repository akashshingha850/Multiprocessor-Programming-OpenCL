#include <stdio.h>
#include <CL/cl.h>

#define MAX_INFO_SIZE 1024

int cuda_info() {
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

int main ()
{
    cuda_info();
}
