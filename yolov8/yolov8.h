

#pragma once
#include "utils/yolo.h"
#include "utils/utils.h"

#include "common/trt_tensor.h"

class YOLOV8 : public YOLO::Yolo {

public:
    YOLOV8(const utils::InitParameter& params);
    ~YOLOV8();
public:
    virtual void task(std::vector<cv::Mat>& imgs_batch);
    virtual void preprocess();
    virtual bool infer();
    virtual void postprocess();

public:
    virtual void preprocess(int ibatch, const cv::Mat& image);
};
