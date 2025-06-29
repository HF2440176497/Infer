

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

    std::vector<float>::iterator min_it = std::min_element(h_src.begin(), h_src.end());
	float min_value = *min_it;

    std::vector<float>::iterator max_it = std::max_element(h_src.begin(), h_src.end());
	float max_value = *max_it;

    std::cout << "min_value: " << min_value << "; max_value: " << max_value << std::endl;

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



void utils::save_float_image_chw(float* d_src, int dst_w, int dst_h, const std::string& save_path, bool normalize) {
    if (!d_src || dst_w <= 0 || dst_h <= 0) {
        throw std::runtime_error("Invalid input parameters");
    }

    size_t data_size = dst_w * dst_h * 3 * sizeof(float);
    std::vector<float> h_src(dst_w * dst_h * 3);

    cudaError_t err = cudaMemcpy(h_src.data(), d_src, data_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
    }

    std::vector<float> h_src_hwc(dst_w * dst_h * 3);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < dst_h; ++y) {  // 行
            for (int x = 0; x < dst_w; ++x) {  // 列
                // CHW: c * (H*W) + y * W + x
                // HWC: (y * W + x) * 3 + c
                h_src_hwc[(y * dst_w + x) * 3 + c] = h_src[c * (dst_h * dst_w) + y * dst_w + x];
            }
        }
    }

    cv::Mat img_float(dst_h, dst_w, CV_32FC3, h_src_hwc.data());
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


/**
 * 转换 CHW 图像 HWC，支持 BGR 通道排列
 */
cv::Mat utils::floatCHW_BGR_to_Mat(const float* data, int width, int height, bool normalize) {
    const int channels = 3;
    std::vector<cv::Mat> chw_channels;

    // 
    for (int c = 0; c < channels; ++c) {
        cv::Mat channel(height, width, CV_32FC1, const_cast<float*>(data + c * height * width));
        chw_channels.push_back(channel);
    }

    cv::Mat mat_float;
    cv::merge(chw_channels, mat_float); // 合并后为 HWC，BGR顺序

    cv::Mat mat_uint8;
    if (normalize) {
        mat_float.convertTo(mat_uint8, CV_8UC3, 255.0);
    } else {
        mat_float.convertTo(mat_uint8, CV_8UC3);
    }
    return mat_uint8;
}

