#pragma once

#include "common/trt_tensor.h"
#include "common_include.h"
#include "utils.h"


/**
 * 负责单张图像的仿射变换
 */
class AffineTrans {
public:
    AffineTrans();
    virtual ~AffineTrans();

public:
    void compute(const std::tuple<int, int>& from, const std::tuple<int, int>& to);  // 相当于初始化

    utils::AffineMat get_d2s() { return m_d2s; }
    utils::AffineMat get_s2d() { return m_s2d; }

public:
    utils::AffineMat m_d2s;
    utils::AffineMat m_s2d;
};

/**
 * 前后处理模块
 */
class Processor {
    enum class ChannelType : int { None = 0, SwapRB = 1 };

public:
    Processor();
    virtual ~Processor();

public:
    void set_stream(cudaStream_t stream, bool owner_stream = false);
    cudaStream_t get_stream();
    void pre_compute(const cv::Mat& image, std::shared_ptr<TRT::Tensor> pre_buffer,
                     std::shared_ptr<TRT::Tensor> net_input);
    void post_compute(int ibatch, std::shared_ptr<TRT::Tensor> net_output, std::shared_ptr<TRT::Tensor> post_buffer,
                      int num_bboxes, int num_classes, int output_cdim, float confidence_threshold, int max_objects);

private:
    void resize_dev(std::shared_ptr<TRT::Tensor> pre_buffer, std::shared_ptr<TRT::Tensor> net_input);
    void channel_swap_dev(std::shared_ptr<TRT::Tensor> net_input, utils::ChannelsArrange order);
    void norm_dev(std::shared_ptr<TRT::Tensor> net_input, utils::ChannelsArrange order);

public:
    int src_w = -1;
    int src_h = -1;
    int dst_w = -1;
    int dst_h = -1;

private:
    AffineTrans m_trans;
    utils::Norm normalize_;
    cudaStream_t stream_;
    bool owner_stream_ = false;
};


