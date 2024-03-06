//kernels.cl
__kernel void resize_image(__read_only image2d_t srcImg, __write_only image2d_t dstImg, sampler_t sampler) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float2 coordSrc = (float2)(coord.x * 4, coord.y * 4);
    uint4 pixel = read_imageui(srcImg, sampler, coordSrc);
    write_imageui(dstImg, coord, pixel);
}

__kernel void grayscale_image(__global const uchar4* input, __global uchar* output, const int width) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int i = y * width + x;
    uchar4 pixel = input[i];
    output[i] = (uchar)(0.2126f * pixel.x + 0.7152f * pixel.y + 0.0722f * pixel.z);
}

__kernel void apply_filter(__global const uchar* input, __global uchar* output, const int width, const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= 2 && y >= 2 && x < (width - 2) && y < (height - 2)) {
        uint sum = 0;
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                sum += input[(y + dy) * width + (x + dx)];
            }
        }
        output[y * width + x] = sum / 25;
    }
}
