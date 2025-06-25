
#include "utils/yolo.h"
#include "yolov8.h"
#include "utils/preprocess.cuh"



YOLOV8::YOLOV8(const utils::InitParameter& params): YOLO::Yolo(params) {
    p_pre = std::make_shared<PreProcess>(params, this->m_input_src_dev, this->m_input_hwc_dev);
}

YOLOV8::~YOLOV8() {

}

/**
 * equivalent to 'forwards'
 */
void YOLOV8::task(std::vector<cv::Mat>& imgs_batch) {
    int image_num = imgs_batch.size();
    this->adjust_mem(image_num);

    for (int i = 0; i < image_num; ++i) {
        preprocess(i, imgs_batch[i]);
    }


}


void YOLOV8::preprocess() {
    p_pre->compute();  // TODO: preprocess compute 作为接口
}


/**
 * @param net_input 预处理结果
 */
void YOLOV8::preprocess(int ibatch, const cv::Mat& image) {

    p_pre->compute(ibatch, image, pre_buffers_, input_buffer_);
}


bool YOLOV8::infer() {


    return true;
}


void YOLOV8::postprocess() {




}



