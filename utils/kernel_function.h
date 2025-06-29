

#pragma once
#include "common_include.h"
#include "utils.h"

#define BLOCK_SIZE 8

#define CHECK(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);

#define checkCudaKernel(...)                                                                         \
    __VA_ARGS__;                                                                                     \
    do{cudaError_t cudaStatus = cudaPeekAtLastError();                                               \
    if (cudaStatus != cudaSuccess){                                                                  \
        INFOE("launch failed: %s", cudaGetErrorString(cudaStatus));                                  \
    }} while(0);


#define INFO(...) __log_func(__FILE__, __LINE__, __VA_ARGS__)
void __log_func(const char *file, int line, const char *fmt, ...);

__global__
void resize_device_kernel_batch(uint8_t* src, int src_w, int src_h, int src_area, int src_volume, 
                        float* dst, int dst_w, int dst_h, int dst_area, int dst_volume, int batch_size,
                        float padding_value, utils::AffineMat matrix);

__global__
void resize_device_kernel(uint8_t* src, int src_w, int src_h, float* dst, int dst_w, int dst_h, 
                        float pad_value, utils::AffineMat matrix);

__device__
void affine_project_device_kernel(utils::AffineMat* matrix, int x, int y, float* proj_x, float* proj_y);
