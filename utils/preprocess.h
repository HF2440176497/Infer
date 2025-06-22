

#include "common_include.h"
#include "preprocessor.h"

/**
 * 负责单张图像的仿射变换
 */
class AffineTrans {
public:
    AffineTrans();
    ~AffineTrans();

public:
    compute(const std::tuple<int, int>& from, 
            const std::tuple<int, int>& to);  // 相当于初始化

private:
    AffineMat m_d2s;
    AffineMat m_s2d;
};


class PreProcess {
    enum class ChannelType : int { None = 0, SwapRB = 1 };

public:
    PreProcess(utils::InitParameter params, float* src, float* dst);
    ~PreProcess();

public:
    void init();
    void set_stream(cudaStream_t stream, bool owner_stream);
    void compute();

private:
    void resize_dev();
    void channel_swap_dev();
    void norm_dev();
    void hwc2chw_dev();

public:
    size_t batch_size;
    int src_w, src_h;
    int dst_w, dst_h;

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