
#pragma once

#include <iostream>
#include <chrono>
#include <string>
#include <sstream>

#include "common_include.h"


namespace utils 
{
    namespace dataSets
    {
        const std::vector<std::string> coco80 = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };
        const std::vector<std::string> coco91 = { 
            "person", "bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
            "fire hydrant","street sign","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
            "hat","backpack","umbrella","shoe","eye glasses","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
            "baseball glove","skateboard","surfboard","tennis racket","bottle","plate","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
            "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","mirror","dining table","window",
            "desk","toilet","door","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","blender",
            "book","clock","vase","scissors","teddy bear","hair drier","toothbrush","hair brush" 
        };
        const std::vector<std::string> voc20 = {
            "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable",
            "dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
        };
    }

    struct InitParameter
    {
        int num_class{ 80 };  // coco 
        std::vector<std::string> class_names;
        std::vector<std::string> input_output_names;

        bool dynamic_batch{ false };
        size_t batch_size;
        int src_h, src_w; 
        int dst_h, dst_w;

        float scale{ 255.f };
        float means[3] = { 0.f, 0.f, 0.f };
        float stds[3] = { 1.f, 1.f, 1.f };

        float iou_thresh;
        float conf_thresh;

        int topK{ 300 };
        std::string save_path;
    };


    struct Box
    {
        float left, top, right, bottom, confidence;
        int label;
        std::vector<cv::Point2i> land_marks;

        Box() = default;
        Box(float left, float top, float right, float bottom, float confidence, int label) :
            left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label) {}

        Box(float left, float top, float right, float bottom, float confidence, int label, int numLandMarks) :
            left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label) 
        {
            land_marks.reserve(numLandMarks);
        }
    };


    struct AffineMat {
        float v0, v1, v2;
        float v3, v4, v5;
        
        __host__ __device__ AffineMat() : v0(1), v1(0), v2(0), v3(0), v4(1), v5(0) {}
        
        __host__ void from_cvmat(const cv::Mat& mat) {
            if (mat.rows != 2 || mat.cols != 3 || mat.type() != CV_32FC1) {
                std::cerr << "AffineMat: Input matrix must be 2x3 and CV_32FC1" << std::endl;
            }
            // std::cout << "mat ptr: " << mat.ptr<float>(0)[0] << std::endl;

            v0 = mat.ptr<float>(0)[0];
            v1 = mat.ptr<float>(0)[1];
            v2 = mat.ptr<float>(0)[2];
            v3 = mat.ptr<float>(1)[0];
            v4 = mat.ptr<float>(1)[1];
            v5 = mat.ptr<float>(1)[2];
        }

        void print(const std::string& title = "AffineMat") const {
            std::printf("%s:\n", title.c_str());
            std::printf("[%10.4f, %10.4f, %10.4f]\n", v0, v1, v2);
            std::printf("[%10.4f, %10.4f, %10.4f]\n", v3, v4, v5);
        }
    };

    enum class ChannelsArrange : int { RGB = 0, BGR = 1 };

    enum class NormType : int { None = 0, MeanStd = 1, AlphaBeta = 2 };

    struct Norm {
        float mean[3];
        float std[3];
        float alpha, beta;
        NormType type = NormType::None;

        // out = (x * alpha - mean) / std
        static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f);

        // out = x * alpha + beta
        static Norm alpha_beta(float alpha, float beta = 0);
        static Norm None();
    };

    std::vector<cv::Mat> load_images(const std::string& folderPath);
    std::vector<uint8_t> load_model(const std::string& file);
    int64_t timestamp_ms();
    std::string file_name(const std::string& path, bool include_suffix);

    void save_float_image(float* d_src, int width, int height, const std::string& save_path, bool normalize = false);
    void save_float_image_chw(float* d_src, int width, int height, const std::string& save_path, 
                            ChannelsArrange order, bool normalize = false);
}  // namespace utils