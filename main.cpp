
#include <stdio.h>
#include <string>
#include "common/ilog.h"
#include "builder/trt_builder.hpp"


int app_yolo();

static void compile(); 

int main(int argc, char** argv) {
    // compile();
    app_yolo();
    return 0;
}

void compile() {

    TRT::Mode mode{TRT::Mode::FP32};
    int batch_size = 8;

    std::string onnx_file = "../model/yolov8s_dy.onnx";
    std::string model_file = "../model/yolov8s_dy.engine";

    TRT::compile(mode,        // FP32、FP16、INT8
                 batch_size,  // max batch size
                 onnx_file,   // source
                 model_file   // save to
    );
}

