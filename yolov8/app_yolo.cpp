
#include <vector>
#include <string>

#include "utils/utils.h"
#include "utils/common_include.h"

#include "yolov8.h"


/**
 * 预设配置
 */
void setParameters(utils::InitParameter& initParameters) {
    initParameters.class_names = utils::dataSets::coco80;
    initParameters.num_class = 80;  // for coco
    initParameters.batch_size = 8;
    initParameters.dst_h = 640;
    initParameters.dst_w = 640;
    initParameters.input_output_names = {"images", "output"};
    initParameters.conf_thresh = 0.25f;
    initParameters.iou_thresh = 0.45f;
    initParameters.save_path = "result";
}


int main(int argc, char** argv) {

    utils::InitParameter params;
    setParameters(params);

    params.src_h = 2048;
    params.src_w = 2048;

    std::vector<cv::Mat> imgs_batch = utils::load_images("../test_imgs");

    int dev = 0;
    CHECK(cudaGetDevice(&dev));
    printf("ID of current CUDA device:  %d\n", dev);
    CHECK(cudaSetDevice(dev));

    std::shared_ptr<YOLO::Yolo> p_yolo = std::make_shared<YOLOV8>(params);

    std::string model_path = "../model/test.model";
    std::vector<uint8_t> trt_file = utils::load_model(model_path);
	if (trt_file.empty()) {
		std::cerr << "trt_file is empty" << std::endl;
		return -1;
	}
    if (p_yolo->init(trt_file) == false) {
		std::cerr << "init engine ocur errors " << std::endl;
		return -1;
    }
    p_yolo->task(imgs_batch);

}
