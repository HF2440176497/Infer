
#include <tuple>

#include "preprocess.cuh"
#include "kernel_function.h"


AffineTrans::AffineTrans() { }

AffineTrans::~AffineTrans() { }


/**
 * @brief 计算矩阵
 */
void AffineTrans::compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to) {

    auto src_w = std::get<0>(from);
    auto src_h = std::get<1>(from);

    auto dst_w = std::get<0>(to);
    auto dst_h = std::get<1>(to);

    float scale_x = (float)(dst_w) / (float)(src_w);
    float scale_y = (float)(dst_h) / (float)(src_h);
    float scale = std::min(scale_x, scale_y);

    cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * src_w + dst_w + scale - 1) * 0.5,
        0.f, scale, (-scale * src_h + dst_h + scale - 1) * 0.5);
    cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);

    cv::invertAffineTransform(src2dst, dst2src);

    m_s2d.from_cvmat(src2dst);
    m_d2s.from_cvmat(dst2src);
}


/**
 * @brief 分配中间变量内存
 */
PreProcess::PreProcess(utils::InitParameter params, uint8_t* src, float* dst) {
    this->batch_size = params.batch_size;
    this->src_w = params.src_w;
    this->src_h = params.src_h;
    this->dst_w = params.dst_w;
    this->dst_h = params.dst_h;

    CHECK(cudaMalloc(&m_resize_dev, batch_size * 3 * dst_h * dst_w * sizeof(float)));
    CHECK(cudaMalloc(&m_swap_dev, batch_size * 3 * dst_h * dst_w * sizeof(float)));
    CHECK(cudaMalloc(&m_norm_dev, batch_size * 3 * dst_h * dst_w * sizeof(float)));

    m_src_dev = src;
    m_hwc_dev = dst;
}


PreProcess::~PreProcess() {
    CHECK(cudaFree(m_resize_dev));
    CHECK(cudaFree(m_swap_dev));
    CHECK(cudaFree(m_norm_dev));
}

/**
 * 初始化仿射变换相关
 */
void PreProcess::init() {
    std::tuple<int, int> from{src_w, src_h};
    std::tuple<int, int> to{dst_w, dst_h};
    m_trans.compute(from, to);
}


/**
 * TODO: 验证预处理
 */
void PreProcess::compute() {
    resize_dev();
    utils::save_float_image(m_resize_dev, dst_w, dst_h, "output.png");
}

/**
 * @param image 原始图像
 * @param net_input 保存前处理结果 用于模型输入
 * @details 拷贝输入图像到 pre_buffer
 */
void PreProcess::compute(int ibatch, const cv::Mat& image,
                        std::vector<std::shared_ptr<TRT::MixMemory>> pre_buffers
                        std::shared_ptr<TRT::Tensor> net_input) {
    if (dst_h == -1 || dst_w == -1) {  // init dst scale
        dst_h = net_input->height();
        dst_w = net_input->width();
    }
    src_w = image.cols;
    src_h = image.rows;
    size_t image_size = src_w * src_w * 3;

    pre_buffers[ibatch].gpu(image_size);
    pre_buffers[ibatch].copy_from_cpu(0, image.data, image_size);

    std::tuple<int, int> from{src_w, src_h};
    std::tuple<int, int> to{dst_w, dst_h};
    m_trans.compute(from, to);

    resize_dev(ibatch, pre_buffers[ibatch], net_input);
}

void PreProcess::set_stream(cudaStream_t stream, bool owner_stream) {
    if (owner_stream_ && stream_) {
        CHECK(cudaStreamDestroy(stream_));
    }
    stream_ = stream;
    owner_stream_ = owner_stream;
}

/**
 * @brief resize and padding
 */
void PreProcess::resize_dev() {
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((dst_w * dst_h + BLOCK_SIZE - 1) / BLOCK_SIZE, (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int src_volume = 3 * src_h * src_w;
    int src_area = src_h * src_w;

    int dst_volume = 3 * dst_h * dst_w;
    int dst_area = dst_h * dst_w;

    float pad_value = 114;

    m_trans.get_d2s().print("d2s");
    m_trans.get_s2d().print("s2d");
    
    resize_device_kernel_batch <<< grid_size, block_size, 0, stream_>>> (
        m_src_dev, src_w, src_h, src_area, src_volume,
        m_resize_dev, dst_w, dst_h, dst_area, dst_volume,
        batch_size, pad_value, m_trans.get_d2s());

        
}

/**
 * 单张图片进行预处理
 */
void PreProcess::resize_dev(int ibatch, std::shared_ptr<TRT::MixMemory> pre_buffer, 
                            std::shared_ptr<TRT::Tensor> net_input) {

    auto current_offset = net_input->offset(ibatch);
    INFO("current offset location [%lld]", current_offset);
    float* dst_dev = net_input->gpu<float>() + current_offset;
    uint8_t* src_dev = pre_buffer.gpu();

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((dst_w + BLOCK_SIZE - 1) / BLOCK_SIZE, (dst_h + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float pad_value = 114;

    m_trans.get_d2s().print("d2s");
    m_trans.get_s2d().print("s2d");
    
    resize_device_kernel <<< grid_size, block_size, 0, stream_>>> (
        src_dev, src_w, src_h, 
        dst_dev, dst_w, dst_h,
        pad_value, m_trans.get_d2s());
}


/**
 * 交换通道，目前是交换
 */
void PreProcess::channel_swap_dev() {



}


void PreProcess::norm_dev() {


}


void PreProcess::hwc2chw_dev() {


}