
#include "utils/yolo.h"
#include "yolov8.h"
#include "utils/preprocess.h"



YOLOV8::YOLOV8(const utils::InitParameter params): YOLO::Yolo(params) {
    p_pre = std::make_shared<PreProcess>(params, this->m_input_src_device, this->m_input_hwc_dev);
}

YOLOV8::~YOLOV8() {

}

void YOLOV8::task(std::vector<cv::Mat>& imgs_batch) {
    this->copy(imgs_batch);
    this->preprocess();
}


void YOLOV8::preprocess() {
    p_pre->compute();  // TODO: preprocess compute 作为接口
}


bool YOLOV8::infer() {



}


void YOLOV8::postprocess() {




}



