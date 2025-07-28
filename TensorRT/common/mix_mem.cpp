
#include "mix_mem.h"


namespace TRT {

	int data_type_size(TRT::DataType dt) {
		switch (dt) {
			case TRT::DataType::Float: return sizeof(float);
            case TRT::DataType::Float16: return sizeof(__half);
			case TRT::DataType::Int32: return sizeof(int);
            case TRT::DataType::Int8: return sizeof(int8_t);
			case TRT::DataType::UInt8: return sizeof(uint8_t);
			default: {
				INFO("Not support dtype: %d", dt);
				return -1;
			}
		}
	}

    int data_type_size(nvinfer1::DataType dt) {
		switch (dt) {
            case nvinfer1::DataType::kFLOAT: return sizeof(float);
            case nvinfer1::DataType::kHALF: return sizeof(__half);
            case nvinfer1::DataType::kINT32: return sizeof(int);
            case nvinfer1::DataType::kINT8: return sizeof(int8_t);
            case nvinfer1::DataType::kUINT8: return sizeof(uint8_t);
			default: {
				INFO("Not support dtype: %d", dt);
				return -1;
			}
		}
	}

    const char* data_type_string(TRT::DataType dt) {
		switch(dt){
			case TRT::DataType::Float: return "Float32";
			case TRT::DataType::Float16: return "Float16";
			case TRT::DataType::Int32: return "Int32";
            case TRT::DataType::Int8: return "Int8";
			case TRT::DataType::UInt8: return "UInt8";
			default: return "Unknow";
		}
	}

    const char* data_type_string(nvinfer1::DataType dt) {
        switch (dt) {
            case nvinfer1::DataType::kFLOAT: return "kFLOAT";
            case nvinfer1::DataType::kHALF: return "kHALF";
            case nvinfer1::DataType::kINT32: return "kINT32";
            case nvinfer1::DataType::kINT8: return "kINT8";
            case nvinfer1::DataType::kUINT8: return "kUINT8";
            default: return "Unknown";
        }
    }

    nvinfer1::DataType to_tensorRT_datatype(TRT::DataType dt) {
        switch (dt) {
            case TRT::DataType::Float: return nvinfer1::DataType::kFLOAT;
            case TRT::DataType::Float16: return nvinfer1::DataType::kHALF;
            case TRT::DataType::Int32: return nvinfer1::DataType::kINT32;
            case TRT::DataType::Int8:  return nvinfer1::DataType::kINT8;
            case TRT::DataType::UInt8: return nvinfer1::DataType::kUINT8;
            default: throw std::runtime_error("Unsupported type for TensorRT");
        }
    }
     
    TRT::DataType to_tensor_datatype(nvinfer1::DataType dt) {
        switch (dt) {
            case nvinfer1::DataType::kFLOAT: return TRT::DataType::Float;
            case nvinfer1::DataType::kHALF: return TRT::DataType::Float16;
            case nvinfer1::DataType::kINT32: return TRT::DataType::Int32;
            case nvinfer1::DataType::kINT8: return TRT::DataType::Int8;
            case nvinfer1::DataType::kUINT8: return TRT::DataType::UInt8;
            default: throw std::runtime_error("Unsupported TensorRT type");
        }
    }

    int get_device(int device_id) {
        if(device_id != CURRENT_DEVICE_ID) {
            CUDATools::check_device_id(device_id);
            return device_id;
        }
        CHECK(cudaGetDevice(&device_id));
        return device_id;
    }

    MixMemory::MixMemory(int device_id){
        device_id_ = get_device(device_id);
    }

    MixMemory::MixMemory(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size){
        reference_data(cpu, cpu_size, gpu, gpu_size);		
    }

    void MixMemory::reference_data(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size){
        release_all();
        
        if(cpu == nullptr || cpu_size == 0){
            cpu = nullptr;
            cpu_size = 0;
        }

        if(gpu == nullptr || gpu_size == 0){
            gpu = nullptr;
            gpu_size = 0;
        }

        this->cpu_ = cpu;
        this->cpu_size_ = cpu_size;
        this->gpu_ = gpu;
        this->gpu_size_ = gpu_size;

        this->owner_cpu_ = !(cpu && cpu_size > 0);
        this->owner_gpu_ = !(gpu && gpu_size > 0);
        CHECK(cudaGetDevice(&device_id_));
    }

    MixMemory::~MixMemory() {
        release_all();
    }

    /**
     * 自动检测是否需要扩容
     */
    void* MixMemory::gpu(size_t size) {
        if (gpu_size_ < size) {
            release_gpu();
            gpu_size_ = size;
            CHECK(cudaMalloc(&gpu_, size));
            CHECK(cudaMemset(gpu_, 0, size));
        }
        return gpu_;
    }

    void* MixMemory::cpu(size_t size) {
        if (cpu_size_ < size) {
            release_cpu();
            cpu_size_ = size;
            CHECK(cudaMallocHost(&cpu_, size));
            Assert(cpu_ != nullptr);
            memset(cpu_, 0, size);
        }
        return cpu_;
    }

    void MixMemory::release_cpu() {
        if (cpu_) {
            if (owner_cpu_) {
                CHECK(cudaFreeHost(cpu_));
            }
            cpu_ = nullptr;
        }
        cpu_size_ = 0;
    }

    void MixMemory::release_gpu() {
        if (gpu_) {
            if (owner_gpu_) {
                CHECK(cudaFree(gpu_));
            }
            gpu_ = nullptr;
        }
        gpu_size_ = 0;
    }

    void MixMemory::release_all() {
        release_cpu();
        release_gpu();
    }

}