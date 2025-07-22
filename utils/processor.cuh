#pragma once

#include "common/trt_tensor.h"
#include "common_include.h"
#include "utils.h"



/**
 * 前后处理模块
 */
class Processor {
    enum class ChannelType : int { None = 0, SwapRB = 1 };

public:
    Processor();
    virtual ~Processor();

public:
    void         set_stream(cudaStream_t stream, bool owner_stream = false);
    cudaStream_t get_stream();
    void pre_compute(const cv::Mat& image, std::shared_ptr<TRT::Tensor> net_input, std::shared_ptr<utils::AffineTrans> trans);
    void post_compute(int ibatch, std::shared_ptr<TRT::Tensor> net_output, std::shared_ptr<TRT::Tensor> post_buffer,
                      int num_bboxes, int num_classes, int output_cdim, float confidence_threshold, int max_objects, std::shared_ptr<utils::AffineTrans> trans);
    void nms_decode(int ibatch, std::shared_ptr<TRT::Tensor> post_buffer, float nms_threshold, int max_objects);

private:
    void resize_dev(std::shared_ptr<TRT::Tensor> pre_buffer, std::shared_ptr<TRT::Tensor> net_input,
                    std::shared_ptr<utils::AffineTrans> trans);
    void channel_swap_dev(std::shared_ptr<TRT::Tensor> net_input, utils::ChannelsArrange order);
    void norm_dev(std::shared_ptr<TRT::Tensor> net_input, utils::ChannelsArrange order);

public:
    int src_w = -1;
    int src_h = -1;
    int dst_w = -1;
    int dst_h = -1;

private:
    utils::Norm normalize_;
    cudaStream_t stream_;
    bool owner_stream_ = false;
    std::shared_ptr<TRT::Tensor>    resize_buffer_ = nullptr;
};


