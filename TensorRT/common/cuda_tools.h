
#pragma once

#include "common_include.h"
#include "kernel_function.h"

const int GPU_BLOCK_THREADS = 256;

#define Assert(op)                        \
    do {                                  \
        bool cond = !(!(op));             \
        if (!cond) {                      \
            INFO("Assert failed, " #op); \
        }                                 \
    } while (false)

namespace CUDATools {

	bool check_device_id(int device_id);
    int current_device_id();
    dim3 grid_dims(int numJobs);
    dim3 block_dims(int numJobs);
    std::string device_capability(int device_id);
    std::string device_name(int device_id);
    std::string device_description();

}
