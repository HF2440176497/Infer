
#include "yolo.h"
#include "common/trt_tensor.h"
#include "common_include.h"

YOLO::Yolo::Yolo(const utils::InitParameter& params): m_param(params) {
    CHECK(cudaMalloc(&m_input_src_dev,    params.batch_size * 3 * params.src_h * params.src_w * sizeof(uint8_t)));
    CHECK(cudaMalloc(&m_input_hwc_dev,    params.batch_size * 3 * params.dst_h * params.dst_w * sizeof(float)));
}

YOLO::Yolo::~Yolo() {
    p_pre = nullptr;
    CHECK(cudaFree(m_input_src_dev));
    CHECK(cudaFree(m_input_hwc_dev));
    if (stream_) {
        CHECK(cudaStreamDestroy(stream_));
    }
}

/**
 * 初始化模型推理的相关
 */
bool YOLO::Yolo::init(const std::vector<uint8_t>& trt_file) {
    CHECK(cudaStreamCreate(&stream_));
    p_pre->set_stream(stream_);
    // p_pre->init();
    return true;
}


/**
 * 将图像拷贝到已分配的显存
 * TODO: 现在已经不需要
 */
void YOLO::Yolo::copy(const std::vector<cv::Mat>& imgs_batch) {
    uint8_t* pi = m_input_src_dev;
    int image_size = sizeof(uint8_t) * m_param.src_w * m_param.src_h * 3;
    for (int i = 0; i < imgs_batch.size(); ++i) {
        CHECK(cudaMemcpy(pi, (uint8_t*)imgs_batch[i].data, image_size, cudaMemcpyHostToDevice));
        pi += m_param.src_w * m_param.src_h * 3;
    }
}


/**
 * 动态调整队列尺寸
 * @param batch_size 实际尺寸
 */
void YOLO::Yolo::adjust_mem(int batch_size) {
    pre_buffers_.clear();
    if ((int)pre_buffers_.size() < batch_size) {
        for (int i = pre_buffers_.size(); i < batch_size; ++i) {
            pre_buffers_.push_back(std::make_shared<TRT::MixMemory>());
        }
    }
    // TODO: 动态设置尺寸
    std::vector<int> dims {batch_size, 3, m_param.dst_h, m_param.dst_w};
    input_buffer_ = std::make_shared<TRT::Tensor>(dims);  // 初始状态
}

/**
 * 检查模型的相关输入输出信息
 */
void YOLO::Yolo::check() {



    
}



