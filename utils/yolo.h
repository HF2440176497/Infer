

#pragma once
#include "common_include.h"
#include "utils.h"
#include "kernel_function.h"

#include "preprocess.h"


namespace YOLO {

    class Yolo {
    public:
        Yolo(const utils::InitParameter& params);
        ~Yolo();
    public:
        virtual void task() == 0;  // main entrance
        virtual void preprocess() = 0;
        virtual bool infer() = 0;
        virtual void postprocess() = 0;

    public:
        virtual bool init(const std::vector<uint8_t>& trt_file);
        virtual void check();
        virtual void copy(const std::vector<CV::Mat>& images_batch);

    protected:
        std::shared_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
        std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
        cudaStream_t stream_ = nullptr;
		std::shared_ptr<IRuntime> runtime_ = nullptr;

    protected:
        uint8_t* m_input_src_dev = nullptr;
        float* m_input_hwc_dev = nullptr;
    
    protected:
        utils::InitParameter m_param;
        std::shared_ptr<PreProcess> p_pre = nullptr;
    };


        

    





}