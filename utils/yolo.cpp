
#include "yolo.h"


YOLO::Yolo::Yolo(const utils::InitParameter& params): m_param(params) {
    CHECK(cudaMalloc(&m_input_src_device,    param.batch_size * 3 * param.src_h * param.src_w * sizeof(uint8_t)));
    CHECK(cudaMalloc(&m_input_hwc_device,    param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
}

YOLO::Yolo::~Yolo() {
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
    p_pre->init();
    return true;
}


/**
 * 将图像拷贝到已分配的显存
 */
void YOLO::Yolo::copy(const std::vector<CV::Mat>& imgs_batch) {
    uint8_t* pi = m_input_src_dev;
    int image_size = sizeof(uint8_t) * m_param.src_w * m_param.src_h * 3;

    for (int i = 0; i < imgs_batch.size(); ++i) {
        CHECK(cudaMemcpy(pi, imgs_batch[i].data, image_size, cudaMemcpyHostToDevice));
        pi += m_param.src_w * m_param.src_h * 3;
    }
}

/**
 * 检查模型的相关输入输出信息
 */
YOLO::Yolo::check() {



    
}
