
#include "common_include.h"
#include "utils.h"

namespace fs = std::filesystem;


/**
 * 加载指定目录下的图片
 */
std::vector<cv::Mat> utils::load_images(const std::string& folder_path) {
    std::vector<cv::Mat> images;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp" || ext == ".tiff") {
                cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);
                if (img.empty()) {
                    std::cerr << "Failed to load image: " << filePath << std::endl;
                    continue;
                }
                images.push_back(img);
            }
        }
    }
    return images;
}

/**
 * @brief 将设备端的 float* 图像数据保存为文件
 * @param d_src 设备端图像数据指针（float*，大小为 dst_w × dst_h × 3）
 * @param dst_w 图像宽度
 * @param dst_h 图像高度
 * @param save_path 保存路径
 * @param normalize 是否将数据从 [0,1] 归一化到 [0,255]
 */
void utils::save_float_image(float* d_src, int dst_w, int dst_h, const std::string& save_path, bool normalize) {
    if (!d_src || dst_w <= 0 || dst_h <= 0) {
        throw std::runtime_error("Invalid input parameters");
    }

    size_t data_size = dst_w * dst_h * 3 * sizeof(float);
    std::vector<float> h_src(dst_w * dst_h * 3);

    cudaError_t err = cudaMemcpy(h_src.data(), d_src, data_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
    }
    cv::Mat img_float(dst_h, dst_w, CV_32FC3, h_src.data());
    cv::Mat img_uint8;

    if (normalize) {
        img_float.convertTo(img_uint8, CV_8UC3, 255.0);
    } else {
        img_float.convertTo(img_uint8, CV_8UC3);
    }
    // 若输入是 RGB，需转换为 BGR
    // cv::cvtColor(img_uint8, img_uint8, cv::COLOR_RGB2BGR);  
    if (!cv::imwrite(save_path, img_uint8)) {
        throw std::runtime_error("Failed to save image: " + save_path);
    }
}



std::vector<uint8_t> utils::load_model(const std::string& file) {
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
    {
        return {};
    }
    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0)
    {
        in.seekg(0, std::ios::beg);
        data.resize(length);
        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}
