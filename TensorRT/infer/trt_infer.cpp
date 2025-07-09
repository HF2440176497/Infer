

#include "trt_infer.h"
#include "utils.h"

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
    if (ptr) ptr->destroy();
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

    void build_model(const void* pdata, size_t size) {
        destroy();
        if (pdata == nullptr || size == 0) return false;

        owner_stream_ = true;
        checkCudaRuntime(cudaStreamCreate(&stream_));
        if (stream_ == nullptr) return false;

        runtime_ = std::shared_ptr<IRuntime>(createInferRuntime(gLogger), destroy_nvidia_pointer<IRuntime>);
        if (runtime_ == nullptr) return false;

        engine_ = std::shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size, nullptr),
                                          destroy_nvidia_pointer<ICudaEngine>);
        if (engine_ == nullptr) return false;

        context_ =
            shared_ptr<IExecutionContext>(engine_->createExecutionContext(), destroy_nvidia_pointer<IExecutionContext>);
        return context_ != nullptr;
    }

    void destory() {
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
    virtual void destory();
    virtual bool load(const std::string& file);
    virtual bool load_from_memory(const void* pdata, size_t size);
    virtual std::shared_ptr<std::vector<uint8_t>> InferImpl::serial_engine();

private:
    void build_engine_input_and_outputs_map();

private:
    int                                  device_ = 0;
    std::shared_ptr<EngineContext>       context_;
    std::vector<std::shared_ptr<Tensor>> inputs_;
    std::vector<std::shared_ptr<Tensor>> outputs_;
    std::vector<int>                     all_ordered_index_;
    std::vector<int>                     inputs_ordered_index_;
    std::vector<int>                     outputs_ordered_index_;
    std::vector<std::string>             inputs_name_;
    std::vector<std::string>             outputs_name_;
    std::vector<std::shared_ptr<Tensor>>              bind_ordered_tensor_;
    std::map<std::string, int>                        bind_name_index_map_;  // bind_name - index
    std::unordered_map<std::string, TensorShapeRange> inputs_shape_map_;
    bool                                              has_static_input_batch_ = false;  // 是否存在
    bool                                              has_dynamic_input_batch_ = false;
};


InferImpl::~InferImpl() {
    destory();
}


InferImpl::destory() {
    int old_device = 0;  // 调用时的原设备
    checkCudaRuntime(cudaGetDevice(&old_device));
    checkCudaRuntime(cudaSetDevice(device_));  // 当前 infer 实例绑定的设备
    this->context_.reset();
    this->inputs_.clear();
    this->outputs_.clear();
    this->inputs_ordered_index_.clear();
    this->outputs_ordered_index_.clear();
    this->inputs_name_.clear();
    this->outputs_name_.clear();
    this->bind_ordered_tensor_.clear();
    this->bind_name_index_map_.clear();
    checkCudaRuntime(cudaSetDevice(old_device));
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
    cudaGetDevice(&device_);
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
    auto memory = this->context_->engine_->serialize();
    auto output = std::make_shared<std::vector<uint8_t>>(
        (uint8_t*)memory->data(), (uint8_t*)memory->data() + memory->size()
    );
    memory->destroy();
    return output;
}

/**
 * 根据模型信息保存映射
 */
void InferImpl::build_engine_input_and_outputs_map() {
    this->context_.reset();
    this->inputs_.clear();
    this->outputs_.clear();
    this->inputs_ordered_index_.clear();
    this->outputs_ordered_index_.clear();
    this->inputs_name_.clear();
    this->outputs_name_.clear();
    this->bind_ordered_tensor_.clear();
    this->bind_name_index_map_.clear();
    auto bind_num = context_->engine_->getNbIOTensors();

    // 保存映射关系
    for (int i = 0; i < bind_num; ++i) {
        auto const bind_name = context_->engine_->getIOTensorName(i);
        TensorIOMode io_mode = context_->engine_->getTensorIOMode(bind_name);
        if (io_mode == TensorIOMode::kINPUT) {
            inputs_name_.push_back(bind_name);
            inputs_ordered_index_.push_back(all_ordered_index_.size());
            
        } else if (io_mode == TensorIOMode::kOUTPUT) {
            outputs_name_.push_back(bind_name);
            outputs_ordered_index_.push_back(all_ordered_index_.size());
        }
        bind_name_index_map[bind_name] = i;
        all_ordered_index_.push_back(i);
    }
    auto opt_profie_index = context_->context_->getOptimizationProfile();
    for (int i = 0; i < inputs_name_.size(); ++i) {  // input bind_name
        auto const bind_name = inputs_name_[i];
        auto       dims = context_->engine_->getTensorShape(bind_name);

        if (dims.d[0] == -1) {
            nvinfer1::Dims min_dims =
                context_->engine_->getProfileShape(bind_name, opt_profie_index, nvinfer1::OptProfileSelector::kMIN);
            nvinfer1::Dims opt_dims =
                context_->engine_->getProfileShape(bind_name, opt_profie_index, nvinfer1::OptProfileSelector::kOPT);
            nvinfer1::Dims max_dims =
                context_->engine_->getProfileShape(bind_name, opt_profie_index, nvinfer1::OptProfileSelector::kMAX);
            inputs_shape_map_[bind_name] = TensorShapeRange(min_dims, opt_dims, max_dims);
        } else {
            inputs_shape_map_[bind_name] = TensorShapeRange(dims);
        }
    }
    auto max_batch_size = get_max_batch_size();
    if (get_max_batch_size == -1) {
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

        // For Input
        if (inputs_shape_map_.find(bind_name) != inputs_shape_map_.end()) {
            inputs_.push_back(new_tensor);
        } else {
            outputs_.push_back(new_tensor);
        }
        bind_ordered_tensor_.push_back(new_tensor);
    }
}


virtual void InferImpl::forward(bool sync) override {
    if (sync) {
        synchronize();
    }
}

/**
 * 判断是否为动态批处理，验证绑定的 input 并返回最大批数量
 * @return -1 未绑定输入映射
 */
virtual int InferImpl::get_max_batch_size() override {
    int  staticBatch == -1;
    int  minLowerBound = 0;        // 动态网络最小batch size的最大值
    int  maxUpperBound = INT_MAX;  // 动态网络最大batch size的最小值
    if (inputs_shape_map_.empty()) {
        return -1;
    }
    for (const auto& pair : inputs_shape_map_) {
        const TensorShapeRange& range = pair.second;
        if (!range.is_dynamic()) {
            has_static_input_batch_ = true;
            int currentBatch = range.fixDims_.d[0];
            if (staticBatch == -1) {
                staticBatch = currentBatch;  // 初始化
            } else if (staticBatch != currentBatch) {
                throw std::runtime_error("Static networks have different batch sizes");
            }
        }
    }
    for (const auto& pair : shapeRanges) {
        const TensorShapeRange& range = pair.second;
        if (range.is_dynamic()) {  // 动态网络
            has_dynamic_input_batch_ = true;
            int minBatch = range.minDims_.d[0];
            int maxBatch = range.maxDims_.d[0];
            if (minBatch > minLowerBound) minLowerBound = minBatch;
            if (maxBatch < maxUpperBound) maxUpperBound = maxBatch;
        }
    }
    // 如果同时存在静态和动态，那么检查静态范围并返回
    if (has_static_input_batch_) {
        if (staticBatch < minLowerBound || staticBatch > maxUpperBound) {
            throw std::runtime_error("Static batch size is outside dynamic range");
        }
        return staticBatch;
    } else {
        if (minLowerBound > maxUpperBound) {
            throw std::runtime_error("Dynamic networks have incompatible ranges");
        }
        return maxUpperBound;
    }
}

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


virtual void InferImpl::set_stream(cudaStream_t stream) override {
    this->context_->set_stream(stream);

    for(auto& t : orderdBlobs_)
        t->set_stream(stream);
}

virtual cudaStream_t InferImpl::get_stream() override { 
    return context_->stream_; 
}

virtual void InferImpl::synchronize() override { 
    checkCudaRuntime(cudaStreamSynchronize(context_->stream_));
}

virtual bool InferImpl::is_input_name(const std::string& name) override {
    return std::find(inputs_name_.begin(), inputs_name_.end(), name) != inputs_name_.end();
}

virtual bool InferImpl::is_output_name(const std::string& name) override {
    return std::find(outputs_name_.begin(), outputs_name_.end(), name) != outputs_name_.end();
}

virtual int InferImpl::num_input() override { 
    return static_cast<int>(this->inputs_.size()); 
}

virtual int InferImpl::num_output() override { 
    return static_cast<int>(this->outputs_.size()); 
}

virtual void InferImpl::print() override {
    if (!context_) {
        INFOW("Infer print, nullptr.");
        return;
    }

    INFO("Infer %p detail", this);
    INFO("\tBase device: %s", CUDATools::device_description().c_str());
    INFO("\tOptimization Profiles num: %d", this->context_->engine_->getNbOptimizationProfiles());
    INFO("\tMax Batch Size: %d", this->get_max_batch_size());
    INFO("\tInputs: %d", inputs_.size());
    for (int i = 0; i < inputs_.size(); ++i) {
        auto& tensor = inputs_[i];
        auto& name = inputs_name_[i];
        INFO("\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
    }

    INFO("\tOutputs: %d", outputs_.size());
    for (int i = 0; i < outputs_.size(); ++i) {
        auto& tensor = outputs_[i];
        auto& name = outputs_name_[i];
        INFO("\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
    }
}

virtual int InferImpl::device() override { return device_; }

/**
 * 返回父类指针
 */
std::shared_ptr<Infer> load_infer(const string& file) {
    std::shared_ptr<InferImpl> Infer(new InferImpl());
    if (!Infer->load(file)) Infer.reset();
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
    checkCudaRuntime(cudaMemGetInfo(&info.available, &info.total));
    return info;
}

int get_device_count() {
    int count = 0;
    checkCudaRuntime(cudaGetDeviceCount(&count));
    return count;
}

int get_device() {
    int device = 0;
    checkCudaRuntime(cudaGetDevice(&device));
    return device;
}

void set_device(int device_id) {
    if (device_id == -1) return;
    checkCudaRuntime(cudaSetDevice(device_id));
}

}  // namespace TRT