

#pragma once
#include "../utils/common_include.h"
#include "../utils/utils.h"

#define CHECK(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

#define BLOCK_SIZE 8

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);

__global__
void resize_device_kernel(uint8_t* src, int src_w, int src_h, int src_area, int src_volume, 
                        float* dst, int dst_w, int dst_h, int dst_area, int dst_volume, int batch_size,
                        float padding_value, utils::AffineMat matrix);

__device__
void affine_project_device_kernel(utils::AffineMat* matrix, int x, int y, float* proj_x, float* proj_y);
