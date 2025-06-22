

#pragma once
#include "utils/yolo.h"
#include "utils/utils.h"

class YOLOV8 : public YOLO::Yolo {

public:
    YOLOV8(const utils::InitParameter& params);
    ~YOLOV8();
public:
    virtual void task(std::vector<cv::Mat>& imgs_batch);
    virtual bool init(const std::vector<uint8_t>& trt_file);
    virtual void preprocess();
    virtual bool infer();
    virtual void postprocess();

};
