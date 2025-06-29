#pragma once

#include "common_include.h"
#include "utils.h"
#include "common/trt_tensor.h"

/**
 * 负责单张图像的仿射变换
 */
class AffineTrans {
public:
    AffineTrans();
    ~AffineTrans();

public:
    void compute(const std::tuple<int, int>& from, 
                const std::tuple<int, int>& to);  // 相当于初始化
    
    utils::AffineMat get_d2s() { return m_d2s; }
    utils::AffineMat get_s2d() { return m_s2d; }

public:
    utils::AffineMat m_d2s;
    utils::AffineMat m_s2d;
};


class PreProcess {
    enum class ChannelType : int { None = 0, SwapRB = 1 };

public:
    PreProcess(utils::InitParameter params, uint8_t* src, float* dst);
    ~PreProcess();

public:
    void init();
    void set_stream(cudaStream_t stream, bool owner_stream = false);
    void compute();
    void compute(int ibatch, const cv::Mat& image,
                std::vector<std::shared_ptr<TRT::MixMemory>> pre_buffers,
                std::shared_ptr<TRT::Tensor> net_input);

private:
    void resize_dev();
    void resize_dev(int ibatch, std::shared_ptr<TRT::MixMemory> pre_buffer, std::shared_ptr<TRT::Tensor> net_input);
    void channel_swap_dev();
    void norm_dev();
    void hwc2chw_dev();

public:
    size_t batch_size;
    int src_w = -1;
    int src_h = -1;
    int dst_w = -1;
    int dst_h = -1;

private:
    uint8_t* m_src_dev;
    float* m_hwc_dev;

private:
    float* m_resize_dev;
    float* m_norm_dev;
    float* m_swap_dev;
    AffineTrans m_trans;
    cudaStream_t stream_;
    bool owner_stream_ = false;
};