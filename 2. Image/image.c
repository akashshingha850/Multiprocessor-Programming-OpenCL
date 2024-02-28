#include "lodepng.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <direct.h> // For _getcwd


void ReadImage(const char* filename, unsigned char** image, unsigned* width, unsigned* height) {
    unsigned error = lodepng_decode32_file(image, width, height, filename);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
        exit(1);
    }
}

void ResizeImage(const unsigned char* inputImage, unsigned inputWidth, unsigned inputHeight,
                 unsigned char** outputImage, unsigned* outputWidth, unsigned* outputHeight) {
    *outputWidth = inputWidth / 4;
    *outputHeight = inputHeight / 4;
    *outputImage = (unsigned char*)malloc((*outputWidth) * (*outputHeight) * 4);

    for (unsigned y = 0; y < *outputHeight; y++) {
        for (unsigned x = 0; x < *outputWidth; x++) {
            for (int i = 0; i < 4; i++) {
                (*outputImage)[(y * (*outputWidth) + x) * 4 + i] =
                    inputImage[(y * 4 * inputWidth + x * 4) * 4 + i];
            }
        }
    }
}

void GrayScaleImage(const unsigned char* inputImage, unsigned inputWidth, unsigned inputHeight,
                    unsigned char** outputImage) {
    *outputImage = (unsigned char*)malloc(inputWidth * inputHeight);
    for (unsigned i = 0; i < inputWidth * inputHeight; i++) {
        unsigned char r = inputImage[i * 4];
        unsigned char g = inputImage[i * 4 + 1];
        unsigned char b = inputImage[i * 4 + 2];
        (*outputImage)[i] = (unsigned char)(0.2126 * r + 0.7152 * g + 0.0722 * b);
    }
}

void ApplyFilter(const unsigned char* grayImage, unsigned width, unsigned height,
                 unsigned char** filteredImage) {
    *filteredImage = (unsigned char*)malloc(width * height);
    memset(*filteredImage, 0, width * height); // Initialize filtered image

    // Simple averaging filter (not an accurate Gaussian blur)
    for (unsigned y = 2; y < height - 2; y++) {
        for (unsigned x = 2; x < width - 2; x++) {
            unsigned sum = 0;
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    sum += grayImage[(y + dy) * width + (x + dx)];
                }
            }
            (*filteredImage)[y * width + x] = sum / 25;
        }
    }
}

void WriteImage(const char* filename, const unsigned char* image, unsigned width, unsigned height) {
    unsigned error = lodepng_encode_file(filename, image, width, height, LCT_GREY, 8);
    if (error) {
        printf("Error %u: %s\n", error, lodepng_error_text(error));
    }
}

void ProfileFunction(void (*func)(const unsigned char*, unsigned, unsigned, unsigned char**),
                     const unsigned char* inputImage, unsigned inputWidth, unsigned inputHeight,
                     unsigned char** outputImage, const char* funcName) {
    clock_t start = clock();
    func(inputImage, inputWidth, inputHeight, outputImage);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%s function execution time: %f seconds\n", funcName, time_spent);
}

int main() {
    const char* inputFile = "D:/Mega/OULU/Multiprocessesor Proggramming/Projects/2. Image/im0.png";  //relative file path not working
    unsigned char *image = NULL, *resizedImage = NULL, *grayImage = NULL, *filteredImage = NULL;
    unsigned width, height, resizedWidth = 0, resizedHeight = 0;

    clock_t start, end;
    double cpu_time_used;

    printf("Current working directory: %s\n", _getcwd(NULL, 0));

    // Reading and decoding the image
    start = clock();
    ReadImage(inputFile, &image, &width, &height);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("ReadImage took %f seconds to execute \n", cpu_time_used);

    // Resizing the image
    start = clock();
    ResizeImage(image, width, height, &resizedImage, &resizedWidth, &resizedHeight);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("ResizeImage took %f seconds to execute \n", cpu_time_used);

    // Converting to grayscale
    start = clock();
    GrayScaleImage(resizedImage, resizedWidth, resizedHeight, &grayImage);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("GrayScaleImage took %f seconds to execute \n", cpu_time_used);

    // Applying the filter
    start = clock();
    ApplyFilter(grayImage, resizedWidth, resizedHeight, &filteredImage);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("ApplyFilter took %f seconds to execute \n", cpu_time_used);

    // Writing the resulting image
    start = clock();
    WriteImage("D:/Mega/OULU/Multiprocessesor Proggramming/Projects/2. Image/im0_out.png", filteredImage, resizedWidth, resizedHeight);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("WriteImage took %f seconds to execute \n", cpu_time_used);

    // Cleanup
    free(image);
    free(resizedImage);
    free(grayImage);
    free(filteredImage);

    return 0;
}
