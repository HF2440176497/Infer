
#include "common_include.h"

#include "cuda_tools.h"

namespace CUDATools {

    bool check_device_id(int device_id) {
        int device_count = -1;
        CHECK(cudaGetDeviceCount(&device_count));
        if(device_id < 0 || device_id >= device_count){
            INFO("Invalid device id: %d, count = %d", device_id, device_count);
            return false;
        }
        return true;
    }

    int current_device_id() {
        int device_id = 0;
        CHECK(cudaGetDevice(&device_id));
        return device_id;
    }

    dim3 grid_dims(int numJobs) {
        int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
        return dim3(((numJobs + numBlockThreads - 1) / numBlockThreads));
    }

    dim3 block_dims(int numJobs) {
        return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    }

    std::string device_capability(int device_id) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, device_id));
        std::ostringstream oss;
        oss << prop.major << "." << prop.minor;
        return oss.str();
    }

    std::string device_name(int device_id) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, device_id));
        return prop.name;
    }

    std::string device_description() {
        cudaDeviceProp prop;
        size_t free_mem, total_mem;
        int device_id = 0;
    
        CHECK(cudaGetDevice(&device_id));
        CHECK(cudaGetDeviceProperties(&prop, device_id));
        CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
        std::ostringstream oss;
        oss << "[ID " << device_id << "]"
            << "<" << prop.name << ">"
            << "[arch " << prop.major << "." << prop.minor << "]"
            << "[GMEM " 
            << std::fixed << std::setprecision(2) 
            << free_mem / (1024.0f * 1024.0f * 1024.0f) << " GB/"
            << total_mem / (1024.0f * 1024.0f * 1024.0f) << " GB]";
    
        return oss.str();
    }

}
