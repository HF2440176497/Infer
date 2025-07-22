
#pragma once

#include "common/detect.hpp"
#include "utils/common_include.h"

namespace Yolo {

class Infer {
public:
    virtual std::vector<std::shared_future<ObjDetect::BoxArray>> commits(const std::vector<cv::Mat>& images) = 0;
};

std::shared_ptr<Infer> create_infer(const std::string& engine_file, int gpuid, float confidence_threshold = 0.25f,
                                    float nms_threshold = 0.5f, int max_objects = 512,
                                    bool use_multi_preprocess_stream = false);

}  // namespace Yolo
