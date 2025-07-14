
#pragma once
#include "common_include.h"
#include "common/trt_tensor.h"


namespace TRT {

struct DeviceMemorySummary {
    size_t total;
    size_t available;
};

struct TensorShapeRange {
    nvinfer1::Dims minDims_;
    nvinfer1::Dims optDims_;
    nvinfer1::Dims maxDims_;
    nvinfer1::Dims fixDims_;
    bool           dynamic_batch_ = false;

    TensorShapeRange() = default;

    explicit TensorShapeRange(const nvinfer1::Dims& minDims, const nvinfer1::Dims& optDims,
                              const nvinfer1::Dims& maxDims)
        : minDims_(minDims), optDims_(optDims), maxDims_(maxDims), dynamic_batch_(true) {
        if (minDims_.nbDims == 0 || optDims_.nbDims == 0 || maxDims_.nbDims == 0) {
            throw std::runtime_error("Dynamic shape requires dims to be set.");
        }
    }

    explicit TensorShapeRange(const nvinfer1::Dims& fixDims) : fixDims_(fixDims), dynamic_batch_(false) {
        if (fixDims_.nbDims == 0) {
            throw std::runtime_error("Static shape requires fixDims to be set.");
        }
    }

    bool validate_batchsize(int actualBatchSize) const {
        if (dynamic_batch_) {
            return actualBatchSize >= minDims_.d[0] && actualBatchSize <= maxDims_.d[0];
        } else {
            return actualBatchSize == fixDims_.d[0];
        }
    }

    int get_max_batch_size() const {
        if (dynamic_batch_) {
            return maxDims_.d[0];
        } else {
            return fixDims_.d[0];
        }
    }

    bool is_dynamic() const { return dynamic_batch_; }
};

class Infer {
public:
    virtual void forward(bool sync = true) = 0;
    virtual int  get_max_batch_size() = 0;

    virtual std::shared_ptr<Tensor> get_input() = 0;
    virtual std::shared_ptr<Tensor> get_output() = 0;
    virtual void                    set_stream(cudaStream_t stream) = 0;
    virtual cudaStream_t            get_stream() = 0;
    virtual void                    synchronize() = 0;
    virtual size_t                  get_device_memory_size() = 0;

    virtual void print() = 0;
    virtual int  device() = 0;
};  // class Infer

DeviceMemorySummary    get_current_device_summary();
int                    get_device_count();
int                    get_device();

void                   set_device(int device_id);
std::shared_ptr<Infer> load_infer_from_memory(const void* pdata, size_t size);
std::shared_ptr<Infer> load_infer(const std::string& file);

}  // namespace TRT
