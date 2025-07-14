

#include "trt_infer.h"
#include "common/trt_tensor.h"
#include "utils/utils.h"

using namespace nvinfer1;

class Logger : public ILogger {
public:
    virtual void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kINTERNAL_ERROR) {
            INFO("NVInfer INTERNAL_ERROR: %s", msg);
            abort();
        } else if (severity == Severity::kERROR) {
            INFO("NVInfer: %s", msg);
        } else if (severity == Severity::kWARNING) {
            INFO("NVInfer: %s", msg);
        } else if (severity == Severity::kINFO) {
            INFO("NVInfer: %s", msg);
        } else {
            INFO("%s", msg);
        }
    }
};

static Logger gLogger;

namespace TRT {

template<typename _T>
static void destroy_nvidia_pointer(_T* ptr) {
    if (ptr) delete ptr;
}


class EngineContext {
public:
    virtual ~EngineContext() { }

    void set_stream(cudaStream_t stream) {
        if (owner_stream_) {
            if (stream_) {
                cudaStreamDestroy(stream_);
            }
            owner_stream_ = false;
        }
        stream_ = stream;
    }

    bool build_model(const void* pdata, size_t size) {
        destroy();
        if (pdata == nullptr || size == 0) return false;

        owner_stream_ = true;
        CHECK(cudaStreamCreate(&stream_));
        if (stream_ == nullptr) return false;

        runtime_ = std::shared_ptr<IRuntime>(createInferRuntime(gLogger), destroy_nvidia_pointer<IRuntime>);
        if (runtime_ == nullptr) return false;

        // before release runtime_, all ICudaEngine instance should be destroyed
        engine_ = std::shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size),
                                          destroy_nvidia_pointer<ICudaEngine>);
        if (engine_ == nullptr) return false;

        context_ =
            std::shared_ptr<IExecutionContext>(engine_->createExecutionContext(), destroy_nvidia_pointer<IExecutionContext>);
        return context_ != nullptr;
    }

    void destroy() {
        context_.reset();
        engine_.reset();
        runtime_.reset();
        if (owner_stream_) {
            if (stream_) {
                cudaStreamDestroy(stream_);
            }
        }
        stream_ = nullptr;
    }

public:
    cudaStream_t stream_ = nullptr;
    bool owner_stream_ = false;
    std::shared_ptr<IRuntime> runtime_ = nullptr;
    std::shared_ptr<ICudaEngine> engine_ = nullptr;
    std::shared_ptr<IExecutionContext> context_ = nullptr;
};

class InferImpl: public Infer {

public:
    virtual ~InferImpl();
    virtual void destroy();
    virtual bool load(const std::string& file);
    virtual bool load_from_memory(const void* pdata, size_t size);
    virtual std::shared_ptr<std::vector<uint8_t>> serial_engine();
    virtual bool validate_batch_size(int actual_size);

private:
    void build_engine_input_and_outputs_map();

private:
    int                                  device_ = 0;
    std::shared_ptr<EngineContext>       context_;
    std::shared_ptr<Tensor>              inputs_;
    std::shared_ptr<Tensor>              outputs_;
    std::vector<int>                     all_ordered_index_;
    int                                  inputs_ordered_index_;  // save input bind index
    int                                  outputs_ordered_index_;
    std::string                          inputs_name_;
    std::string                          outputs_name_;
    std::vector<std::shared_ptr<Tensor>>              bind_ordered_tensor_;
    std::map<std::string, int>                        bind_name_index_map_;  // bind_name - index
    TensorShapeRange                                  inputs_shape_;
    bool                                              has_static_input_batch_ = false;  // 是否存在
    bool                                              has_dynamic_input_batch_ = false;

// override
public:
    virtual void                    forward(bool sync = true) override;
    virtual int                     get_max_batch_size() override;
    virtual std::shared_ptr<Tensor> get_input() override;
    virtual std::shared_ptr<Tensor> get_output() override;
    virtual void                    set_stream(cudaStream_t stream) override;
    virtual cudaStream_t            get_stream() override;
    virtual void                    synchronize() override;
    virtual void                    print() override;
    virtual int                     device() override;
    virtual size_t                  get_device_memory_size() override;
};

InferImpl::~InferImpl() {
    destroy();
}

void InferImpl::destroy() {
    int old_device = 0;  // 调用时的原设备
    CHECK(cudaGetDevice(&old_device));
    CHECK(cudaSetDevice(device_));  // 当前 infer 实例绑定的设备
    this->context_.reset();
    all_ordered_index_.clear();
    bind_ordered_tensor_.clear();
    CHECK(cudaSetDevice(old_device));
}

/**
 * 读取模型文件
 */
bool InferImpl::load(const std::string& file) {
    auto data = utils::load_model(file);
    if (data.empty()) return false;
    context_.reset(new EngineContext());
    if (!context_->build_model(data.data(), data.size())) {
        context_.reset();
        return false;
    }
    cudaGetDevice(&device_);  // 先前由外部设置，此处读取保存为成员变量
    build_engine_input_and_outputs_map();
    return true;
}

bool InferImpl::load_from_memory(const void* pdata, size_t size) {
    if (pdata == nullptr || size == 0) return false;

    context_.reset(new EngineContext());

    // build model
    if (!context_->build_model(pdata, size)) {
        context_.reset();
        return false;
    }
    cudaGetDevice(&device_);
    build_engine_input_and_outputs_map();
    return true;
}

std::shared_ptr<std::vector<uint8_t>> InferImpl::serial_engine() {
    IHostMemory* memory = this->context_->engine_->serialize();
    auto output = std::make_shared<std::vector<uint8_t>>(
        (uint8_t*)memory->data(), (uint8_t*)memory->data() + memory->size()
    );
    delete memory;
    return output;
}

/**
 * 根据模型信息保存映射
 */
void InferImpl::build_engine_input_and_outputs_map() {
    this->inputs_.reset();
    this->outputs_.reset();
    this->all_ordered_index_.clear();
    this->bind_ordered_tensor_.clear();
    this->bind_name_index_map_.clear();
    auto bind_num = context_->engine_->getNbIOTensors();

    // 保存映射关系
    for (int i = 0; i < bind_num; ++i) {
        auto const bind_name = context_->engine_->getIOTensorName(i);
        TensorIOMode io_mode = context_->engine_->getTensorIOMode(bind_name);
        if (io_mode == TensorIOMode::kINPUT) {
            inputs_name_ = bind_name;
            inputs_ordered_index_ = all_ordered_index_.size();
        } else if (io_mode == TensorIOMode::kOUTPUT) {
            outputs_name_ = bind_name;
            outputs_ordered_index_ = all_ordered_index_.size();
        }
        bind_name_index_map_[bind_name] = i;
        all_ordered_index_.push_back(i);
    }
    auto opt_profie_index = context_->context_->getOptimizationProfile();

    if (all_ordered_index_.size() != 2) {
        INFO("Model should has one input and one output %d", all_ordered_index_.size());
        return;
    }
    // analysis model input shape
    auto dims = context_->engine_->getTensorShape(inputs_name_.c_str());
    if (dims.d[0] == -1) {
        nvinfer1::Dims min_dims =
            context_->engine_->getProfileShape(inputs_name_.c_str(), opt_profie_index, nvinfer1::OptProfileSelector::kMIN);
        nvinfer1::Dims opt_dims =
            context_->engine_->getProfileShape(inputs_name_.c_str(), opt_profie_index, nvinfer1::OptProfileSelector::kOPT);
        nvinfer1::Dims max_dims =
            context_->engine_->getProfileShape(inputs_name_.c_str(), opt_profie_index, nvinfer1::OptProfileSelector::kMAX);
        inputs_shape_ = TensorShapeRange(min_dims, opt_dims, max_dims);
    } else {  // static shape
        inputs_shape_ = TensorShapeRange(dims);
    }
    auto max_batch_size = get_max_batch_size();
    if (max_batch_size == -1) {
        INFO("get_max_batch_size null");
        return;
    }
    for (int i = 0; i < bind_num; ++i) {
        auto const bind_name = context_->engine_->getIOTensorName(i);
        auto dims = context_->engine_->getTensorShape(bind_name);
        dims.d[0] = max_batch_size;  // suitable size
        auto data_type = context_->engine_->getTensorDataType(bind_name);
        auto format = context_->engine_->getTensorFormat(bind_name);

        auto new_tensor = std::make_shared<TRT::Tensor>(dims, data_type, format);
        new_tensor->set_stream(this->context_->stream_);

        if (bind_name == inputs_name_) {
            inputs_ = new_tensor;
        } else {
            outputs_ = new_tensor;
        }
        bind_ordered_tensor_.push_back(new_tensor);
    }
}


void InferImpl::forward(bool sync) {




    if (sync) {
        synchronize();
    }
}


std::shared_ptr<Tensor> InferImpl::get_input() {
    return inputs_;
}

std::shared_ptr<Tensor> InferImpl::get_output() {
    return outputs_;
}

/**
 * 判断是否为动态批处理，验证绑定的 input 并返回最大批数量
 * @return -1 未绑定输入映射
 */
int InferImpl::get_max_batch_size() {
    int  staticBatch = -1;
    if (!inputs_shape_.is_dynamic()) {
        has_static_input_batch_ = true;
        int currentBatch = inputs_shape_.fixDims_.d[0];
        if (staticBatch == -1) {
            staticBatch = currentBatch;  // 初始化
        } else if (staticBatch != currentBatch) {
            throw std::runtime_error("Static networks have different batch sizes");
        }
        return staticBatch;
    }
    has_dynamic_input_batch_ = true;
    int minBatch = inputs_shape_.minDims_.d[0];
    int maxBatch = inputs_shape_.maxDims_.d[0];
    return maxBatch;
}

// virtual int InferImpl::get_max_batch_size() override {
//     int  staticBatch == -1;
//     int  minLowerBound = 0;        // 动态网络最小batch size的最大值
//     int  maxUpperBound = INT_MAX;  // 动态网络最大batch size的最小值
//     if (inputs_shape_map_.empty()) {
//         return -1;
//     }
//     for (const auto& pair : inputs_shape_map_) {
//         const TensorShapeRange& range = pair.second;
//         if (!range.is_dynamic()) {
//             has_static_input_batch_ = true;
//             int currentBatch = range.fixDims_.d[0];
//             if (staticBatch == -1) {
//                 staticBatch = currentBatch;  // 初始化
//             } else if (staticBatch != currentBatch) {
//                 throw std::runtime_error("Static networks have different batch sizes");
//             }
//         }
//     }
//     for (const auto& pair : shapeRanges) {
//         const TensorShapeRange& range = pair.second;
//         if (range.is_dynamic()) {  // 动态网络
//             has_dynamic_input_batch_ = true;
//             int minBatch = range.minDims_.d[0];
//             int maxBatch = range.maxDims_.d[0];
//             if (minBatch > minLowerBound) minLowerBound = minBatch;
//             if (maxBatch < maxUpperBound) maxUpperBound = maxBatch;
//         }
//     }
//     // 如果同时存在静态和动态，那么检查静态范围并返回
//     if (has_static_input_batch_) {
//         if (staticBatch < minLowerBound || staticBatch > maxUpperBound) {
//             throw std::runtime_error("Static batch size is outside dynamic range");
//         }
//         return staticBatch;
//     } else {
//         if (minLowerBound > maxUpperBound) {
//             throw std::runtime_error("Dynamic networks have incompatible ranges");
//         }
//         return maxUpperBound;
//     }
// }

/**
 * 验证传入的实际 batch_size 是否符合模型要求
 */
bool InferImpl::validate_batch_size(int actual_size) {
    int suitable_size = get_max_batch_size();
    if (has_static_input_batch_) {
        if (suitable_size != actual_size) {
            return false;
        }
    } else {
        if (actual_size > suitable_size) {
            return false;
        }
    }
    return true;
}


void InferImpl::set_stream(cudaStream_t stream) {
    this->context_->set_stream(stream);

    for(auto& t : bind_ordered_tensor_)
        t->set_stream(stream);
}

cudaStream_t InferImpl::get_stream() { 
    return context_->stream_; 
}

void InferImpl::synchronize() { 
    CHECK(cudaStreamSynchronize(context_->stream_));
}


void InferImpl::print() {
    if (!context_) {
        INFO("Infer print, nullptr.");
        return;
    }
    INFO("Infer %p detail", this);
    INFO("\tBase device: %s", CUDATools::device_description().c_str());
    // INFO("\tOptimization Profiles num: %d", this->context_->engine_->getNbOptimizationProfiles());
    INFO("\tMax Batch Size: %d", this->get_max_batch_size());
    INFO("\tInput bind name: %s : shape {%s}, %s", inputs_name_.c_str(), inputs_->shape_string(), data_type_string(inputs_->type()));
    INFO("\tOutput bind name: %s : shape {%s}, %s", outputs_name_.c_str(), outputs_->shape_string(), data_type_string(outputs_->type()));
    
}

size_t InferImpl::get_device_memory_size() {
    return context_->context_->getEngine().getDeviceMemorySizeV2();  // getEngine return ICudaEngine
}

int InferImpl::device() { return device_; }

/**
 * 返回父类指针
 */
std::shared_ptr<Infer> load_infer(const std::string& file) {
    std::shared_ptr<InferImpl> Infer(new InferImpl());
    if (!Infer->load(file)) {
        INFO("Infer Load failed");
        Infer.reset();
    }
    return Infer;
}

std::shared_ptr<Infer> load_infer_from_memory(const void* pdata, size_t size){
    std::shared_ptr<InferImpl> Infer(new InferImpl());
    if (!Infer->load_from_memory(pdata, size))
        Infer.reset();
    return Infer;
}

DeviceMemorySummary get_current_device_summary() {
    DeviceMemorySummary info;
    CHECK(cudaMemGetInfo(&info.available, &info.total));
    return info;
}

int get_device_count() {
    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    return count;
}

int get_device() {
    int device = 0;
    CHECK(cudaGetDevice(&device));
    return device;
}

void set_device(int device_id) {
    if (device_id == -1) return;
    CHECK(cudaSetDevice(device_id));
}

}  // namespace TRT