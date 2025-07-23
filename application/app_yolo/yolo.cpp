

#include "yolo.h"
#include "infer/trt_infer.h"
#include "common/InferController.h"
#include "common/trt_tensor.h"
#include "utils/processor.cuh"
#include "common/detect.hpp"
#include "utils/kernel_function.h"

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

public:
    virtual bool startup(const std::string& file, int device_id, float confidence_threshold, float nms_threshold,
                         int max_objects, bool use_multi_preprocess_stream);

private:
    int                             input_width_ = 0;
    int                             input_height_ = 0;
    int                             max_batch_size_ = 0;
    float                           confidence_threshold_ = 0;
    float                           nms_threshold_ = 0;
    int                             max_objects_ = 512;
    cudaStream_t                    stream_ = nullptr;
    bool                            use_multi_preprocess_stream_ = false;
    std::shared_ptr<Processor>      proc_ = nullptr;
    std::shared_ptr<TRT::Tensor>    output_array_device_ = nullptr;  // 后处理输出
                                                                     // dim: (infer_batch_size, 1 + max_objects_ * NUM_BOX_ELEMENT)
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
    if (proc_ == nullptr) {
        INFO("proc_ is null");
        return false;
    }
    job.input = image;
    job.trans = std::make_shared<utils::AffineTrans>();
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
            proc_->set_stream(preprocess_stream, true);
        } else {
            preprocess_stream = stream_;
            proc_->set_stream(preprocess_stream, false);
        }
    }
    tensor->set_stream(preprocess_stream);
    tensor->resize(1, 3, input_height_, input_width_);  // 用于保存预处理的结果, 应当是 NCHW 排列
    proc_->pre_compute(job.input, tensor, job.trans);
    return true;
}

std::vector<std::shared_future<ObjDetect::BoxArray>> InferImpl::commits(const std::vector<cv::Mat>& images) {
    return ControllerImpl::commits(images);
}

void InferImpl::worker(std::promise<bool>& result) {
    auto file = start_param_.model_file();
    int device_id = start_param_.device_id();
    TRT::set_device(device_id);

    auto engine = TRT::load_infer(file);
    if (engine == nullptr) {
        INFO("Engine load failed");
        result.set_value(false);
        return;
    }
    stream_ = engine->get_stream();  // the stream used to infer
    auto input = engine->get_input();
    auto output = engine->get_output();

    input_width_ = input->width();
    input_height_ = input->height();
    
    max_batch_size_ = engine->get_max_batch_size();  // ps: input dims[0] already max_batch_size 
    tensor_allocator_ = std::make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size_ * 2);
    proc_ = std::make_shared<Processor>();
    output_array_device_ = std::make_shared<TRT::Tensor>(output->type());

    jobs_queue_ = std::make_unique<JobQueue<Job>>(
        max_batch_size_ * 3,  // max_size
        max_batch_size_ * 2,  // warn_size
        [](size_t size) { INFO("WARNING: job_queue size: %d", size); },
        [](Job& job) { if (job.pro) job.pro->set_value(typename ControllerImpl::OutputType()); }
    );

    engine->print();
    result.set_value(true);  // 到此处是同步逻辑，不会引起竞争
    std::vector<Job> fetch_jobs{};
    while (run_ && jobs_queue_->get_jobs_and_wait(fetch_jobs, max_batch_size_)) {
        
        int infer_batch_size = fetch_jobs.size();  // actual_batch
        INFO("max_batch_size: %d; infer_batch_size: %d", max_batch_size_, infer_batch_size);

        if (!engine->validate_batch_size(infer_batch_size)) {
            INFO("ERROR: actual batch size not valid");
            return;
        }
        // 设置实际批长度
        input->resize_single_dim(0, infer_batch_size);
        output->resize_single_dim(0, infer_batch_size);
        output->to_gpu(false);

        if (output_array_device_ == nullptr) {
            INFO("ERROR: output_array_device_ not allocate");
            return;
        }
        output_array_device_->resize(infer_batch_size, 1 + max_objects_ * NUM_BOX_ELEMENT);
        output_array_device_->to_gpu(false);

        for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
            auto& job = fetch_jobs[ibatch];
            auto& mono = job.mono_tensor->data();

            // make sure preprocess sync
            if (proc_->get_stream() != stream_) {
                CHECK(cudaStreamSynchronize(proc_->get_stream()));
            }
            if (mono->get_stream() != proc_->get_stream()) {
                CHECK(cudaStreamSynchronize(mono->get_stream()));
            }
            input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());  // mono already in device: preprocess
            job.mono_tensor->release();
        }
        engine->forward(true);

        int num_bboxes = output->size(2);
        int output_cdim = output->size(1);
        int num_classes = output_cdim - 4;

        for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
            auto& job = fetch_jobs[ibatch];
            proc_->post_compute(ibatch, output, output_array_device_, num_bboxes, num_classes, output_cdim,
                                confidence_threshold_, max_objects_, job.trans);
            proc_->nms_decode(ibatch, output_array_device_, nms_threshold_, max_objects_);
        }
        proc_->synchronize();
        output_array_device_->to_cpu(true);  // copy to cpu
        for (int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
            float* parray = output_array_device_->cpu<float>(ibatch);
            int    count = std::min(max_objects_, static_cast<int>(parray[0]));
            auto&  job = fetch_jobs[ibatch];  // from jobs_
            auto&  image_based_boxes = job.output;
            for (int i = 0; i < count; ++i) {
                float* pbox  = parray + 1 + i * NUM_BOX_ELEMENT;
                int label    = pbox[5];
                int keepflag = pbox[6];
                if (keepflag == 1) {
                    image_based_boxes.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                }
            }
            job.pro->set_value(image_based_boxes);
        }
    }  // end while()
    stream_ = nullptr;
    tensor_allocator_.reset();
    INFO("worker exit");
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
