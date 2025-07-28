
#include "common_include.h"
#include "trt_tensor.h"


namespace TRT {

    Tensor::Tensor(nvinfer1::DataType dtype, std::shared_ptr<MixMemory> data, int device_id) {
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
    Tensor::Tensor(nvinfer1::Dims dims, nvinfer1::DataType dtype, nvinfer1::TensorFormat format,
                   std::shared_ptr<MixMemory> data, int device_id) {
        this->dtype_ = dtype;
        this->device_id_ = get_device(device_id);
        descriptor_string_[0] = 0;
        setup_data(data);
        if (format != nvinfer1::TensorFormat::kLINEAR) {
            std::cerr << "Not supported format" << std::endl;
            return;
        }
        std::vector<int> dims_vec;
        for (int i = 0; i < dims.nbDims; ++i) {
            dims_vec.push_back(dims.d[i]);
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
        return -1;
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
        return -1;
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
        return -1;
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

		shape_string_[0] = 0;

		char* buffer = shape_string_;
		size_t buffer_size = sizeof(shape_string_);
		for(int i = 0; i < shape_.size(); ++i){

			int size = 0;
			if (i < shape_.size() - 1)
				size = snprintf(buffer, buffer_size, "%d x ", shape_[i]);
			else
				size = snprintf(buffer, buffer_size, "%d", shape_[i]);

			buffer += size;
			buffer_size -= size;
		}
		return *this;
	}

    /**
     * data_ 会重新释放再引用
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
        updated_ = DataHead::Init;
        if (data_->cpu()) {
            head_ = head_ | DataHead::Host;
        }
        if (data_->gpu()) {
            head_ = head_ | DataHead::Device;
        }
    }

    Tensor& Tensor::copy_from_gpu(size_t offset, const void* src, size_t num_element, int device_id) {
        if (head_ == DataHead::Init || updated_ == DataHead::Init) to_gpu();

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

        // 按照更新标记拷贝内存
        if (updated_ == DataHead::Device) {
            int current_device_id = get_device(device_id);
            int gpu_device_id = device();
            if (current_device_id != gpu_device_id) {
                CHECK(cudaMemcpyPeerAsync(gpu<uint8_t>() + offset_location, gpu_device_id, src,
                                                     current_device_id, copyed_bytes, stream_));
            } else {
                CHECK(cudaMemcpyAsync(gpu<uint8_t>() + offset_location, src, copyed_bytes,
                                                 cudaMemcpyDeviceToDevice, stream_));
            }
        } else if (updated_ == DataHead::Host) {
            CHECK(cudaMemcpyAsync(cpu<uint8_t>() + offset_location, src, copyed_bytes,
                                cudaMemcpyDeviceToHost, stream_));
        } else {
            INFO("Unsupport update_ type %d", updated_);
        }
        return *this;
    }

    /**
     * @param offset 拷贝到目标位置的元素偏移数
     */
    Tensor& Tensor::copy_from_cpu(size_t offset, const void* src, size_t num_element) {
        if (head_ == DataHead::Init || updated_ == DataHead::Init) to_cpu();

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
        // 按照更新标记拷贝内存
        if (updated_ == DataHead::Device) {
            CHECK(cudaMemcpyAsync(gpu<uint8_t>() + offset_location, src, copyed_bytes,
                                             cudaMemcpyHostToDevice, stream_));
        } else if (updated_ == DataHead::Host) {
            memcpy(cpu<uint8_t>() + offset_location, src, copyed_bytes);
        } else {
            INFO("Unsupport update_ type %d", updated_);
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
        if (needed_size > this->bytes_) {  // 需要重新
            head_ = DataHead::Init;
            updated_ = DataHead::Init;
        }
        this->bytes_ = needed_size;
        return *this;
    }

    Tensor& Tensor::synchronize(){ 
		CHECK(cudaStreamSynchronize(stream_));
		return *this;
	}

	Tensor& Tensor::to_gpu() {
		if (updated_ == DataHead::Device)
			return *this;

		head_ = head_ | DataHead::Device;
		data_->gpu(bytes_);

        if (updated_ == DataHead::Host) {
            CHECK(cudaMemcpyAsync(data_->gpu(), data_->cpu(), bytes_, cudaMemcpyHostToDevice, stream_));
            CHECK(cudaStreamSynchronize(stream_));
        }
        updated_ = DataHead::Device;
		return *this;
	}
	
    /**
     * 并不会对原已分配的数据进行检验
     */
	Tensor& Tensor::to_cpu() {
		if (updated_ == DataHead::Host)
			return *this;

		head_ = head_ | DataHead::Host;
		data_->cpu(bytes_);

        if (updated_ == DataHead::Device) {
			CHECK(cudaMemcpyAsync(data_->cpu(), data_->gpu(), bytes_, cudaMemcpyDeviceToHost, stream_));
			CHECK(cudaStreamSynchronize(stream_));
        }
        updated_ = DataHead::Host;
		return *this;
	}

};
