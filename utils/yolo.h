

#pragma once
#include "common_include.h"
#include "utils.h"
#include "kernel_function.h"

#include "preprocess.cuh"
#include "common/trt_tensor.h"


namespace YOLO {

    class Yolo {
    public:
        Yolo(const utils::InitParameter& params);
        ~Yolo();
    public:
        virtual void task(std::vector<cv::Mat>& imgs_batch) = 0;  // main entrance
        virtual void preprocess() = 0;
        virtual bool infer() = 0;
        virtual void postprocess() = 0;

    public:
        virtual bool init(const std::vector<uint8_t>& trt_file);
        virtual void check();
        virtual void copy(const std::vector<cv::Mat>& images_batch);
        virtual void adjust_mem(int batch_size);

    protected:
        std::shared_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
        std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
        cudaStream_t stream_;
		// std::shared_ptr<IRuntime> runtime_ = nullptr;

    protected:
        uint8_t* m_input_src_dev = nullptr;
        float* m_input_hwc_dev = nullptr;
        std::vector<std::shared_ptr<TRT::MixMemory>> pre_buffers_;  // 保存输入图像
        std::shared_ptr<TRT::Tensor> input_buffer_;  // 模型输入 NCHW

    protected:
        utils::InitParameter m_param;
        std::shared_ptr<PreProcess> p_pre = nullptr;
        int input_width_, input_height_;  // 模型要求的输入
        int device_id_ = 0;
    };


        

    
}