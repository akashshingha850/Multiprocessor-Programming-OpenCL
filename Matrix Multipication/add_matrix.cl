__kernel void add_matrix(__global const float* matrix_1,
                         __global const float* matrix_2,
                         __global float* result) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    int rows = get_global_size(0);
    int cols = get_global_size(1);

    result[i * cols + j] = matrix_1[i * cols + j] + matrix_2[i * cols + j];
}
