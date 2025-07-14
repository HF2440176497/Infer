

#include "yolo.h"
#include "infer/trt_infer.h"
#include "common/InferController.h"
#include "common/trt_tensor.h"
#include "utils/preprocess.cuh"
#include "common/detect.hpp"


namespace Yolo {

/**
 * 封装相关操作为类成员
 */
struct StartParamType {
    std::string model_file_;
    int device_id_ = -1;

    StartParamType() = default;
    StartParamType(std::string model_file, int device_id)
        : model_file_(model_file), device_id_(device_id) {}

    const std::string& model_file() const { return model_file_; }
    int device_id() const { return device_id_; }

    void set_model_file(std::string file) { model_file_ = file; }
    void set_device_id(int id) { device_id_ = id; }
};

using ControllerImpl = InferController
<
    cv::Mat,
    ObjDetect::BoxArray,
    StartParamType
>;

class InferImpl : public Infer, public ControllerImpl {
public:
    virtual std::vector<std::shared_future<ObjDetect::BoxArray>> commits(const std::vector<cv::Mat>& images) override;

public:
    virtual bool preprocess(Job& job, const cv::Mat& image) override;
    virtual void worker(std::promise<bool>& result) override;
    virtual bool worker_serial() override;

public:
    virtual void adjust_mem();
    virtual bool startup(const std::string& file, int device_id, float confidence_threshold, float nms_threshold,
                         int max_objects, bool use_multi_preprocess_stream);

private:
    int                             input_width_ = 0;
    int                             input_height_ = 0;
    float                           confidence_threshold_ = 0;
    float                           nms_threshold_ = 0;
    int                             max_objects_ = 1024;
    cudaStream_t                    stream_ = nullptr;
    bool                            use_multi_preprocess_stream_ = false;
    std::shared_ptr<PreProcess>     pre_ = nullptr;
    std::shared_ptr<TRT::MixMemory> pre_buffer_ = nullptr;
};  // class InferImpl

/**
 * Override Controller::preprocess
 */
bool InferImpl::preprocess(Job& job, const cv::Mat& image) {
    if (image.empty()) {
        INFO("Image is empty");
        return false;
    }
    if (tensor_allocator_ == nullptr) {
        INFO("tensor_allocator_ is nullptr");
        return false;
    }
    if (pre_ == nullptr) {
        INFO("pre_ is null");
        return false;
    }
    job.mono_tensor = tensor_allocator_->query();  // std::shared_ptr<MonopolyData>
    if (job.mono_tensor == nullptr) {
        INFO("Tensor allocator query failed.");
        return false;
    }
    auto& tensor = job.mono_tensor->data();  // std::shared_ptr<Tensor>&
    cudaStream_t preprocess_stream = nullptr;
    if (tensor == nullptr) {  // 虽然 datas_ 含有 MonopolyData, 但是初始时 MonopolyData 的 data_ null
        tensor = std::make_shared<TRT::Tensor>();
        if (use_multi_preprocess_stream_) {
            CHECK(cudaStreamCreate(&preprocess_stream));
            pre_->set_stream(preprocess_stream, true);
        } else {
            preprocess_stream = stream_;
            pre_->set_stream(preprocess_stream, false);
        }
    }
    tensor->set_stream(preprocess_stream);
    tensor->resize(1, 3, input_height_, input_width_);  // tensor 用于保存预处理的结果
    pre_->compute(image, pre_buffer_, tensor);
    return true;
}

std::vector<std::shared_future<ObjDetect::BoxArray>> InferImpl::commits(const std::vector<cv::Mat>& images) {
    // return ControllerImpl::commits(images);
    return ControllerImpl::commits_serial(images);
}

void InferImpl::worker(std::promise<bool>& result) {


}

/**
 * TODO: 串行初始化模型 
 */
bool InferImpl::worker_serial() {
    auto file = start_param_.model_file();
    int device_id = start_param_.device_id();
    TRT::set_device(device_id);

    auto engine = TRT::load_infer(file);
    if (engine == nullptr) {
        INFO("Engine load failed");
        return false;
    }
    stream_ = engine->get_stream();  // 推理所用的 stream

    std::shared_ptr<TRT::Tensor> input = engine->get_input();
    std::shared_ptr<TRT::Tensor> output = engine->get_output();
    input_width_ = input->width();
    input_height_ = input->height();
    this->adjust_mem();

    int max_batch_size = engine->get_max_batch_size();  // ps: input dims[0] already max_batch_size 
    tensor_allocator_ = std::make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
    pre_ = std::make_shared<PreProcess>();
    engine->print();
    return true;
}

void InferImpl::adjust_mem() {
    this->pre_buffer_ = std::make_shared<TRT::MixMemory>();
}

// TODO: 参数未来分离为单独模块
bool InferImpl::startup(const std::string& file, int device_id, float confidence_threshold, float nms_threshold,
                                int max_objects, bool use_multi_preprocess_stream) {
    confidence_threshold_ = confidence_threshold;
    nms_threshold_ = nms_threshold;
    max_objects_ = max_objects;
    use_multi_preprocess_stream_ = use_multi_preprocess_stream;
    return ControllerImpl::startup(StartParamType{file, device_id});
}

std::shared_ptr<Infer> create_infer(const std::string& engine_file, int device_id, float confidence_threshold,
                                    float nms_threshold, int max_objects, bool use_multi_preprocess_stream) {
    std::shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(engine_file, device_id, confidence_threshold, nms_threshold, max_objects,
                           use_multi_preprocess_stream)) {
        instance.reset();
    }
    return instance;
}

}  // namespace Yolo
