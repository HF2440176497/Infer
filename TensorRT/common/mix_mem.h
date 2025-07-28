
#pragma once

#include "common_include.h"
#include "cuda_tools.h"

#define CURRENT_DEVICE_ID           -1

namespace TRT {

    enum class DataHead : uint32_t {
        Init   = 0,
        Host   = 1 << 0,
        Device = 1 << 1
    };

    inline DataHead operator|(DataHead a, DataHead b) {
        return static_cast<DataHead>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }

    inline DataHead operator&(DataHead a, DataHead b) {
        return static_cast<DataHead>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
    }

    enum class DataType : int {
        Unknow = -1,
        Float = 0,
        Float16 = 1,
        Int32 = 2,
        Int8 = 3,
        UInt8 = 4
    };

    int data_type_size(DataType dt);
    int data_type_size(nvinfer1::DataType dt);

    const char* data_type_string(TRT::DataType dt);
    const char* data_type_string(nvinfer1::DataType dt);

    nvinfer1::DataType to_tensorRT_datatype(TRT::DataType dt);
    TRT::DataType to_tensor_datatype(nvinfer1::DataType dt);

    int get_device(int device_id);

    class MixMemory {
    public:
        MixMemory(int device_id = CURRENT_DEVICE_ID);
        MixMemory(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size);
        virtual ~MixMemory();
        void* gpu(size_t size);
        void* cpu(size_t size);
        void release_gpu();
        void release_cpu();
        void release_all();
    
        inline bool owner_gpu() const{return owner_gpu_;}
        inline bool owner_cpu() const{return owner_cpu_;}
    
        inline size_t cpu_size() const{return cpu_size_;}
        inline size_t gpu_size() const{return gpu_size_;}
        inline int device_id() const{return device_id_;}
    
        inline void* gpu() const { return gpu_; }
        inline void* cpu() const { return cpu_; }
    
        void reference_data(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size);
    
    private:
        void* cpu_ = nullptr;
        size_t cpu_size_ = 0;
        bool owner_cpu_ = true;
        int device_id_ = 0;
    
        void* gpu_ = nullptr;
        size_t gpu_size_ = 0;
        bool owner_gpu_ = true;
    };

}