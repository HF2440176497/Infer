
#include "common_include.h"
#include "trt_tensor.h"


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

    inline static int get_device(int device_id){
        if(device_id != CURRENT_DEVICE_ID){
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

    MixMemory& MixMemory::copy_from_cpu(size_t offset, const void* src, 
                                        size_t copyed_bytes, cudaStream_t stream) {
        if (offset >= gpu_size_) {
            INFO("Offset location[%lld] >= gpu_size_[%lld], out of range", offset, gpu_size_);
            return *this;
        }
        size_t remain_bytes = gpu_size_ - offset;
        if (copyed_bytes > remain_bytes) {
            INFO("Copyed bytes[%lld] > remain bytes[%lld], out of range", copyed_bytes, remain_bytes);
            return *this;
        }
        CHECK(cudaMemcpyAsync(gpu_ + offset, src, copyed_bytes, cudaMemcpyHostToDevice, stream));
        return *this;
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

    const char* data_head_string(DataHead dh){
        switch(dh){
            case DataHead::Init: return "Init";
            case DataHead::Device: return "Device";
            case DataHead::Host: return "Host";
            default: return "Unknow";
        }
    }

    Tensor::Tensor(nvinfer1::DataType dtype, std::shared_ptr<MixMemory> data, int device_id){
		shape_string_[0] = 0;
		descriptor_string_[0] = 0;
		this->device_id_ = get_device(device_id);
		dtype_ = dtype;
		setup_data(data);
	}

    Tensor::Tensor(int n, int c, int h, int w, nvinfer1::DataType dtype, std::shared_ptr<MixMemory> data, int device_id) {
		this->dtype_ = dtype;
		this->device_id_ = get_device(device_id);
		descriptor_string_[0] = 0;
		setup_data(data);
		resize(n, c, h, w);  // resize(int ndims, const int* dims); dims = [n, c, h, w]
	}

	Tensor::Tensor(int ndims, const int* dims, nvinfer1::DataType dtype, std::shared_ptr<MixMemory> data, int device_id) {
		this->dtype_ = dtype;
		this->device_id_ = get_device(device_id);
		descriptor_string_[0] = 0;
		setup_data(data);
		resize(ndims, dims);
	}

    Tensor::Tensor(const std::vector<int>& dims, nvinfer1::DataType dtype, std::shared_ptr<MixMemory> data, int device_id){
		this->dtype_ = dtype;
		this->device_id_ = get_device(device_id);
		descriptor_string_[0] = 0;
		setup_data(data);
		resize(dims);
	}

    /**
     * 根据模型相关信息 创建需要的维度
     */
    Tensor::Tensor(nvinfer1::Dims dims, nvinfer1::DataType dtype, nvinfer1::TensorFormat format, std::shared_ptr<MixMemory> data, int device_id) {
		this->dtype_ = dtype;
		this->device_id_ = get_device(device_id);
		descriptor_string_[0] = 0;
		setup_data(data);
        std::vector<int> dims_vec {};
        if (format == nvinfer1::TensorFormat::kLINEAR) {
            dims_vec.push_back(dims.d[0]);
            dims_vec.push_back(dims.d[1]);
            dims_vec.push_back(dims.d[2]);
            dims_vec.push_back(dims.d[3]);
        } else if (format == nvinfer1::TensorFormat::kCHW2) {
            dims_vec.push_back(dims.d[0]);
            dims_vec.push_back(div_up(dims.d[1], 2));
            dims_vec.push_back(dims.d[2]);
            dims_vec.push_back(dims.d[3]);
            dims_vec.push_back(2);
        } else if (format == nvinfer1::TensorFormat::kCHW4) {
            dims_vec.push_back(dims.d[0]);
            dims_vec.push_back(div_up(dims.d[1], 4));
            dims_vec.push_back(dims.d[2]);
            dims_vec.push_back(dims.d[3]);
            dims_vec.push_back(4);
        } else if (format == nvinfer1::TensorFormat::kCHW32) {
            dims_vec.push_back(dims.d[0]);
            dims_vec.push_back(div_up(dims.d[1], 32));
            dims_vec.push_back(dims.d[2]);
            dims_vec.push_back(dims.d[3]);
            dims_vec.push_back(32);
        } else if (format == nvinfer1::TensorFormat::kHWC8) {
            dims_vec.push_back(dims.d[0]);
            dims_vec.push_back(dims.d[2]);
            dims_vec.push_back(dims.d[3]);
            dims_vec.push_back(div_up(dims.d[1], 8) * 8);
        }
        resize(dims_vec);
    }

	Tensor::~Tensor() {
		release();
	}

    int Tensor::batch() const {
        return shape_[0];
    }

    // 注意：除了 kLINEAR，其他类型不代表真实宽度、高度
    int Tensor::channel() const {
        switch (format_) {
            case nvinfer1::TensorFormat::kLINEAR:
            case nvinfer1::TensorFormat::kCHW2:
            case nvinfer1::TensorFormat::kCHW4:
            case nvinfer1::TensorFormat::kCHW32:
                return shape_[1];
            case nvinfer1::TensorFormat::kHWC8:
                return shape_[3];
            default:
                INFO("Unsupported tensor format");
        }
    }

    int Tensor::height() const {
        switch (format_) {
            case nvinfer1::TensorFormat::kLINEAR:
            case nvinfer1::TensorFormat::kCHW2:
            case nvinfer1::TensorFormat::kCHW4:
            case nvinfer1::TensorFormat::kCHW32:
                return shape_[2];
            case nvinfer1::TensorFormat::kHWC8:
                return shape_[1];
            default:
                INFO("Unsupported tensor format");
        }
    }

    int Tensor::width() const {
        switch (format_) {
            case nvinfer1::TensorFormat::kLINEAR:
            case nvinfer1::TensorFormat::kCHW2:
            case nvinfer1::TensorFormat::kCHW4:
            case nvinfer1::TensorFormat::kCHW32:
                return shape_[3];
            case nvinfer1::TensorFormat::kHWC8:
                return shape_[2];
            default:
                INFO("Unsupported tensor format");
        }
    }

    const char* Tensor::descriptor() const{
		char* descriptor_ptr = (char*)descriptor_string_;
		int device_id = device();
		snprintf(descriptor_ptr, sizeof(descriptor_string_), 
			"Tensor:%p, %s, %s, CUDA:%d", 
			data_.get(),
			data_type_string(dtype_), 
			shape_string_, 
			device_id
		);
		return descriptor_ptr;
	}

	Tensor& Tensor::compute_shape_string(){

		// clean string
		shape_string_[0] = 0;

		char* buffer = shape_string_;
		size_t buffer_size = sizeof(shape_string_);
		for(int i = 0; i < shape_.size(); ++i){

			int size = 0;
			if(i < shape_.size() - 1)
				size = snprintf(buffer, buffer_size, "%d x ", shape_[i]);
			else
				size = snprintf(buffer, buffer_size, "%d", shape_[i]);

			buffer += size;
			buffer_size -= size;
		}
		return *this;
	}

    /**
     * data_ 会重现释放再引用
     */
	void Tensor::reference_data(const std::vector<int>& shape, void* cpu_data, size_t cpu_size, 
                                void* gpu_data, size_t gpu_size, nvinfer1::DataType dtype) {
		dtype_ = dtype;
		data_->reference_data(cpu_data, cpu_size, gpu_data, gpu_size);
		setup_data(data_);  // 将成员传入
		resize(shape);
	}

    void Tensor::setup_data(std::shared_ptr<MixMemory> data) {
        data_ = data;
        if (data_ == nullptr) {
            data_ = std::make_shared<MixMemory>(device_id_);
        } else {
            device_id_ = data_->device_id();
        }
        head_ = DataHead::Init;
        if (data_->cpu()) {
            head_ = DataHead::Host;
        }
        if (data_->gpu()) {
            head_ = DataHead::Device;
        }
    }

    std::shared_ptr<Tensor> Tensor::clone() const {
        auto new_tensor = std::make_shared<Tensor>(shape_, dtype_);
        if (head_ == DataHead::Init) return new_tensor;

        if (head_ == DataHead::Host) {
            memcpy(new_tensor->cpu(), this->cpu(), this->bytes_);
        } else if (head_ == DataHead::Device) {
            CHECK(
                cudaMemcpyAsync(new_tensor->gpu(), this->gpu(), bytes_, cudaMemcpyDeviceToDevice, stream_));
        }
        return new_tensor;
    }

    Tensor& Tensor::copy_from_gpu(size_t offset, const void* src, size_t num_element, int device_id) {
        if (head_ == DataHead::Init) to_gpu(false);

        size_t offset_location = offset * element_size();
        if (offset_location >= bytes_) {
            INFO("Offset location[%lld] >= bytes_[%lld], out of range", offset_location, bytes_);
            return *this;
        }

        size_t copyed_bytes = num_element * element_size();
        size_t remain_bytes = bytes_ - offset_location;
        if (copyed_bytes > remain_bytes) {
            INFO("Copyed bytes[%lld] > remain bytes[%lld], out of range", copyed_bytes, remain_bytes);
            return *this;
        }

        if (head_ == DataHead::Device) {
            int current_device_id = get_device(device_id);
            int gpu_device_id = device();
            if (current_device_id != gpu_device_id) {
                CHECK(cudaMemcpyPeerAsync(gpu<uint8_t>() + offset_location, gpu_device_id, src,
                                                     current_device_id, copyed_bytes, stream_));
            } else {
                CHECK(cudaMemcpyAsync(gpu<uint8_t>() + offset_location, src, copyed_bytes,
                                                 cudaMemcpyDeviceToDevice, stream_));
            }
        } else if (head_ == DataHead::Host) {
            CHECK(cudaMemcpyAsync(cpu<uint8_t>() + offset_location, src, copyed_bytes,
                                             cudaMemcpyDeviceToHost, stream_));
        } else {
            INFO("Unsupport head type %d", head_);
        }
        return *this;
    }

    /**
     * @param offset 拷贝到目标位置的元素偏移数
     */
    Tensor& Tensor::copy_from_cpu(size_t offset, const void* src, size_t num_element) {
        if (head_ == DataHead::Init) to_cpu(false);

        size_t offset_location = offset * element_size();
        if (offset_location >= bytes_) {
            INFO("Offset location[%lld] >= bytes_[%lld], out of range", offset_location, bytes_);
            return *this;
        }

        size_t copyed_bytes = num_element * element_size();
        size_t remain_bytes = bytes_ - offset_location;
        if (copyed_bytes > remain_bytes) {
            INFO("Copyed bytes[%lld] > remain bytes[%lld], out of range", copyed_bytes, remain_bytes);
            return *this;
        }
        if (head_ == DataHead::Device) {
            CHECK(cudaMemcpyAsync(data_->gpu() + offset_location, src, copyed_bytes,
                                             cudaMemcpyHostToDevice, stream_));
        } else if (head_ == DataHead::Host) {
            memcpy(data_->cpu() + offset_location, src, copyed_bytes);
        } else {
            INFO("Unsupport head type %d", head_);
        }
        return *this;
    }

    Tensor& Tensor::release() {
        data_->release_all();
        shape_.clear();
        bytes_ = 0;
        head_ = DataHead::Init;
        if (stream_owner_ && stream_ != nullptr) {
            CHECK(cudaStreamDestroy(stream_));
        }
        stream_owner_ = false;
        stream_ = nullptr;
        return *this;
    }

    bool Tensor::empty() const{
		return data_->cpu() == nullptr && data_->gpu() == nullptr;
	}

    int Tensor::offset_array(size_t size, const int* index_array) const {
        Assert(size <= shape_.size());
        int value = 0;
        for (int i = 0; i < shape_.size(); ++i) {
            if (i < size) value += index_array[i];

            if (i + 1 < shape_.size()) value *= shape_[i + 1];
        }
        return value;
    }

    int Tensor::offset_array(const std::vector<int>& index_array) const{
		return offset_array(index_array.size(), index_array.data());
	}

    /**
     * 计算元素数量
     */
    int Tensor::count(int start_axis) const {
        if (start_axis >= 0 && start_axis < shape_.size()) {
            int size = 1;
            for (int i = start_axis; i < shape_.size(); ++i) size *= shape_[i];
            return size;
        } else {
            return 0;
        }
    }

    Tensor& Tensor::resize(const std::vector<int>& dims) {
		return resize(dims.size(), dims.data());
	}

    int Tensor::numel() const {
        int value = shape_.empty() ? 0 : 1;
        for (int i = 0; i < shape_.size(); ++i) {
            value *= shape_[i];
        }
        return value;
    }

    Tensor& Tensor::resize_single_dim(int idim, int size){
		Assert(idim >= 0 && idim < shape_.size());
		auto new_shape = shape_;
		new_shape[idim] = size;
		return resize(new_shape);
	}

    /**
     * @details dims -1 表示采用对应的原维度，此时要求 dims.size == shape_.size
     */
    Tensor& Tensor::resize(int ndims, const int* dims) {
        std::vector<int> setup_dims(ndims);
        for (int i = 0; i < ndims; ++i) {
            int dim = dims[i];
            if (dim == -1) {
                Assert(ndims == shape_.size());
                dim = shape_[i];
            }
            setup_dims[i] = dim;
        }
        this->shape_ = setup_dims;
        this->strides_.resize(setup_dims.size());

        size_t prev_size = element_size();
        size_t prev_shape = 1;
        for (int i = (int)strides_.size() - 1; i >= 0; --i) {
            if (i + 1 < strides_.size()) {
                prev_size = strides_[i + 1];
                prev_shape = shape_[i + 1];
            }
            strides_[i] = prev_size * prev_shape;
        }

        this->adajust_memory_by_update_dims_or_type();
        this->compute_shape_string();
        return *this;
    }

    /**
     * @brief 根据 dims 去更新自己的 bytes_ 为 needed_size
     */
    Tensor& Tensor::adajust_memory_by_update_dims_or_type() {
        int needed_size = this->numel() * element_size();
        if (needed_size > this->bytes_) {
            head_ = DataHead::Init;
        }
        this->bytes_ = needed_size;
        return *this;
    }

    Tensor& Tensor::synchronize(){ 
		CHECK(cudaStreamSynchronize(stream_));
		return *this;
	}

	Tensor& Tensor::to_gpu(bool copy) {

		if (head_ == DataHead::Device)
			return *this;

		head_ = DataHead::Device;
		data_->gpu(bytes_);

		if (copy && data_->cpu() != nullptr) {
			CHECK(cudaMemcpyAsync(data_->gpu(), data_->cpu(), bytes_, cudaMemcpyHostToDevice, stream_));
		}
		return *this;
	}
	
	Tensor& Tensor::to_cpu(bool copy) {

		if (head_ == DataHead::Host)
			return *this;

		head_ = DataHead::Host;
		data_->cpu(bytes_);

		if (copy && data_->gpu() != nullptr) {
			CHECK(cudaMemcpyAsync(data_->cpu(), data_->gpu(), bytes_, cudaMemcpyDeviceToHost, stream_));
			CHECK(cudaStreamSynchronize(stream_));
		}
		return *this;
	}

};