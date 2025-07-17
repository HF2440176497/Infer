

#include <tuple>

#include "kernel_function.h"
#include "utils/processor.cuh"


AffineTrans::AffineTrans() { }

AffineTrans::~AffineTrans() { }


/**
 * @brief 计算矩阵
 */
void AffineTrans::compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to) {

    auto src_w = std::get<0>(from);
    auto src_h = std::get<1>(from);

    auto dst_w = std::get<0>(to);
    auto dst_h = std::get<1>(to);

    float scale_x = (float)(dst_w) / (float)(src_w);
    float scale_y = (float)(dst_h) / (float)(src_h);
    float scale = std::min(scale_x, scale_y);

    cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * src_w + dst_w + scale - 1) * 0.5,
        0.f, scale, (-scale * src_h + dst_h + scale - 1) * 0.5);
    cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);

    cv::invertAffineTransform(src2dst, dst2src);

    m_s2d.from_cvmat(src2dst);
    m_d2s.from_cvmat(dst2src);
}


/**
 * @brief 分配中间变量内存
 */
Processor::Processor() {
    normalize_ = utils::Norm::alpha_beta(1 / 255.0f, 0.0f);
}


Processor::~Processor() {
    if (owner_stream_ && stream_) {
        CHECK(cudaStreamDestroy(stream_));
    }
    owner_stream_ = false;
    stream_ = nullptr;
}

/**
 * @param image 原始图像
 * @param net_input 保存前处理结果 用于模型输入
 * @details 拷贝输入图像到 pre_buffer
 */
void Processor::pre_compute(const cv::Mat& image, std::shared_ptr<TRT::Tensor> pre_buffer,
                            std::shared_ptr<TRT::Tensor> net_input) {
    if (dst_h == -1 || dst_w == -1) {  // init dst scale
        dst_h = net_input->height();
        dst_w = net_input->width();
    }
    src_w = image.cols;
    src_h = image.rows;
    size_t image_size = src_w * src_h * 3;  // bytes

    // std::cout << "src_w: " << src_w 
    // << "; src_h: " << src_h 
    // << "; image_size: " << image_size << std::endl;

    pre_buffer->resize(std::vector<int>{1, src_h, src_w, 3});  // For cv::Mat, NHWC
    pre_buffer->to_gpu();
    pre_buffer->copy_from_cpu(0, image.data, image_size);

    std::tuple<int, int> from{src_w, src_h};
    std::tuple<int, int> to{dst_w, dst_h};
    m_trans.compute(from, to);

    resize_dev(pre_buffer, net_input);  // out: CHW BGR
    channel_swap_dev(net_input, utils::ChannelsArrange::BGR);  // out: CHW RGB
    norm_dev(net_input, utils::ChannelsArrange::RGB);  // out: CHW RGB

    int64_t timestamp = utils::timestamp_ms();
    std::string filename = std::to_string(timestamp) + ".png";
    utils::save_float_image_chw(net_input->gpu<float>(), dst_w, dst_h, filename, utils::ChannelsArrange::RGB, true);
}

/**
 * 这里是批处理 因此需要指定 ibatch
 */
void Processor::post_compute(int ibatch, std::shared_ptr<TRT::Tensor> net_output, std::shared_ptr<TRT::Tensor> post_buffer,
                             int num_bboxes, int num_classes, int output_cdim,
                             float confidence_threshold, int max_objects) {
    float* image_based_output = net_output->gpu<float>(ibatch);  // offset = ibatch
    float* output_array_ptr = post_buffer->gpu<float>(ibatch);

    int unit_output_size = 1 + max_objects * NUM_BOX_ELEMENT;
    CHECK(cudaMemsetAsync(output_array_ptr, 0, unit_output_size, stream_));  // initialize

    dim3 grid_size = CUDATools::grid_dims(num_bboxes);
    dim3 block_size = CUDATools::block_dims(num_bboxes);
    checkCudaKernel(decode_kernel<<<grid_size, block_size, 0, stream_>>>(
        image_based_output, output_array_ptr, num_bboxes, num_classes, output_cdim, confidence_threshold, max_objects,
        m_trans.get_d2s()));
}

void Processor::set_stream(cudaStream_t stream, bool owner_stream) {
    if (owner_stream_ && stream_) {
        CHECK(cudaStreamDestroy(stream_));
    }
    stream_ = stream;
    owner_stream_ = owner_stream;
}

cudaStream_t Processor::get_stream() {
    return stream_;
}

/**
 * 单张图片进行预处理
 */
void Processor::resize_dev(std::shared_ptr<TRT::Tensor> pre_buffer, 
                            std::shared_ptr<TRT::Tensor> net_input) {

    float* dst_dev = net_input->gpu<float>();
    uint8_t* src_dev = pre_buffer->gpu<uint8_t>();

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((dst_w + BLOCK_SIZE - 1) / BLOCK_SIZE, (dst_h + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float pad_value = 114;

    // m_trans.get_d2s().print("d2s");
    // m_trans.get_s2d().print("s2d");
    
    resize_device_kernel <<< grid_size, block_size, 0, stream_>>> (
        src_dev, src_w, src_h, 
        dst_dev, dst_w, dst_h,
        pad_value, m_trans.get_d2s());
}


/**
 * 交换通道
 */
void Processor::channel_swap_dev(std::shared_ptr<TRT::Tensor> net_input, utils::ChannelsArrange order) {

    float* dst_dev = net_input->gpu<float>();

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((dst_w + BLOCK_SIZE - 1) / BLOCK_SIZE, (dst_h + BLOCK_SIZE - 1) / BLOCK_SIZE);

    swap_rb_channels_kernel_chw <<< grid_size, block_size, 0, stream_>>> (dst_dev, dst_w, dst_h, order);
}

/**
 * 标准化 指定三通道排列
 */
void Processor::norm_dev(std::shared_ptr<TRT::Tensor> net_input, utils::ChannelsArrange order) {
    float* dst_dev = net_input->gpu<float>();

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((dst_w + BLOCK_SIZE - 1) / BLOCK_SIZE, (dst_h + BLOCK_SIZE - 1) / BLOCK_SIZE);

    normalize_kernel_chw <<< grid_size, block_size, 0, stream_>>> (dst_dev, dst_w, dst_h, normalize_, order);
}
