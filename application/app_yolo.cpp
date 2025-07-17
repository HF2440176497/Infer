

#include "builder/trt_builder.h"
#include "infer/trt_infer.h"
#include "common/detect.hpp"
#include "common/ilog.h"
#include "common/trt_tensor.h"
#include "utils/utils.h"

#include "app_yolo/yolo.h"


static void inference_and_performance(int device_id, std::string model_file) {
    auto engine = Yolo::create_infer(
        model_file,                 // engine file
        device_id,                  // gpu id
        0.25f,                      // confidence threshold
        0.45f,                      // nms threshold
        512,                        // max objects
        false                       // preprocess use multi stream
    );
    if (engine == nullptr) {
        INFO("Engine is null");
        return;
    }

    std::vector<cv::Mat> images = utils::load_images("../data/images", 2);
    std::vector<std::shared_future<ObjDetect::BoxArray>> boxes_array{};

    const int ntest = 1;
    auto begin_time = utils::timestamp_ms();

    for (int i = 0; i < ntest; ++i) {
        boxes_array = engine->commits(images);  // std::vector<std::shared_future<Output>>
        boxes_array.back().wait();
        Assert(boxes_array.size() == images.size());
        INFO("Current ntest index [%d] complete", i);
    }
    int inference_average_time = (utils::timestamp_ms() - begin_time) / ntest;
    INFO("inference average time : %d", inference_average_time);

    // 绘制结果
    for (int i = 0; i < boxes_array.size(); ++i) {
        auto& image = images[i];
        auto boxes  = boxes_array[i].get();
    }

}

static void test(std::string model_file) {

    int device_id = 0;
    TRT::set_device(device_id);

    std::vector<cv::Mat> imgs_batch = utils::load_images("../data/images");
    inference_and_performance(device_id, model_file);
}


int app_yolo() {
    test("../model/yolov8_fp16.engine");
    return 0;
}
