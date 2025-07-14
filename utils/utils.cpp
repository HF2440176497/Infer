

#include "common_include.h"
#include "utils.h"

namespace fs = std::filesystem;


utils::Norm utils::Norm::mean_std(const float mean[3], const float std[3], float alpha) {
    Norm out;
    out.type = NormType::MeanStd;
    out.alpha = alpha;
    memcpy(out.mean, mean, sizeof(out.mean));
    memcpy(out.std, std, sizeof(out.std));
    return out;
}

utils::Norm utils::Norm::alpha_beta(float alpha, float beta) {
    Norm out;
    out.type = NormType::AlphaBeta;
    out.alpha = alpha;
    out.beta = beta;
    return out;
}

utils::Norm utils::Norm::None() { return Norm(); }

/**
 * 加载指定目录下的图片
 */
std::vector<cv::Mat> utils::load_images(const std::string& folder_path, int loop_num) {
    std::vector<cv::Mat> images;

    for (int i = 0; i < loop_num; ++i) {
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
    }  // for (loop_num)
    return images;
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


int64_t utils::timestamp_ms() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();
    return ms;
}


/**
 * 提取路径中的文件名 include_suffix：是否包含后缀
 */
std::string utils::file_name(const std::string& path, bool include_suffix) {
    if (path.empty()) return "";

    int p = path.rfind('/');
    int e = path.rfind('\\');
    p = std::max(p, e);
    p += 1;

    // include suffix
    if (include_suffix) return path.substr(p);

    int u = path.rfind('.');
    if (u == -1) return path.substr(p);

    if (u <= p) u = path.size();
    return path.substr(p, u - p);
}

/**
 * @brief 将设备端的 float* 图像数据保存为文件
 * @param d_src 设备端图像数据指针（float*，大小为 dst_w × dst_h × 3）
 * @param save_path 保存路径
 * @param normalize 是否将数据从 [0,1] 归一化到 [0,255]
 */
void utils::save_float_image(float* d_src, int width, int height, const std::string& save_path, bool normalize) {
    if (!d_src || width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid input parameters");
    }

    size_t data_size = width * height * 3 * sizeof(float);
    std::vector<float> h_src(width * height * 3);

    cudaError_t err = cudaMemcpy(h_src.data(), d_src, data_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
    }

    // std::vector<float>::iterator min_it = std::min_element(h_src.begin(), h_src.end());
    // std::vector<float>::iterator max_it = std::max_element(h_src.begin(), h_src.end());
    // std::cout << "min_value: " << *min_it << "; max_value: " << *max_it << std::endl;

    cv::Mat img_float(height, width, CV_32FC3, h_src.data());
    cv::Mat img_uint8;

    if (normalize) {
        img_float.convertTo(img_uint8, CV_8UC3, 255.0);
    } else {
        img_float.convertTo(img_uint8, CV_8UC3);
    }
    if (!cv::imwrite(save_path, img_uint8)) {
        throw std::runtime_error("Failed to save image: " + save_path);
    }
}


void utils::save_float_image_chw(float* d_src, int width, int height, const std::string& save_path, 
                                ChannelsArrange order, bool normalize) {
    if (!d_src || width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid input parameters");
    }

    size_t data_size = width * height * 3 * sizeof(float);
    std::vector<float> h_src(width * height * 3);

    cudaError_t err = cudaMemcpy(h_src.data(), d_src, data_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
    }

    std::vector<float> h_src_hwc(width * height * 3);
    int target_c;
    for (int c = 0; c < 3; ++c) {
        if (order == ChannelsArrange::RGB) {
            target_c = 2 - c;
        } else if (order == ChannelsArrange::BGR) {
            target_c = c;
        } else { target_c = c; }
        for (int y = 0; y < height; ++y) {  // 行
            for (int x = 0; x < width; ++x) {  // 列
                // CHW: c * (H*W) + y * W + x
                // HWC: (y * W + x) * 3 + c
                h_src_hwc[(y * width + x) * 3 + target_c] = h_src[c * (height * width) + y * width + x];
            }
        }
    }
    cv::Mat img_float(height, width, CV_32FC3, h_src_hwc.data());
    cv::Mat img_uint8;

    if (normalize) {
        cv::normalize(img_float, img_float, 0, 255, cv::NORM_MINMAX);
        img_float.convertTo(img_uint8, CV_8UC3);
    } else {
        img_float.convertTo(img_uint8, CV_8UC3);
    }
    if (!cv::imwrite(save_path, img_uint8)) {
        throw std::runtime_error("Failed to save image: " + save_path);
    }
}
