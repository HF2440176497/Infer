
#include <math.h>

#include "kernel_function.h"
#include "utils.h"


bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
	if (code != cudaSuccess) {
		const char* err_name = cudaGetErrorName(code);
		const char* err_message = cudaGetErrorString(code);
		printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
		return false;
	}
	return true;
}

void __log_func(const char* file, int line, const char* fmt, ...) {
    va_list vl;
    va_start(vl, fmt);
    char buffer[2048];
    std::string filename = utils::file_name(file, true);  // 当前调用处的代码文件
    int n = snprintf(buffer, sizeof(buffer), "[%s:%d]: ", filename.c_str(), line);
    vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
    fprintf(stdout, "%s\n", buffer);
}


/**
 * 批量图片处理
 * @param src_area 像素位置数目
 * @param src_volume 包含通道数的像素数
 * @param martix dst2src 
 * @details 适用于 NCHW 格式（通道分离）
 */
__global__ 
void resize_device_kernel_batch(uint8_t* src, int src_w, int src_h, int src_area, int src_volume, 
						float* dst, int dst_w, int dst_h, int dst_area, int dst_volume, 
						int batch_size, float padding_value, utils::AffineMat matrix) {

	int dx = blockDim.x * blockIdx.x + threadIdx.x;  // 目标图像像素处理位置
	int dy = blockDim.y * blockIdx.y + threadIdx.y;  // batch 处理位置

	if (dx < dst_area && dy < batch_size) {
		int dst_y = dx / dst_w;  // 行维度
		int dst_x = dx % dst_w;  // 列维度

		float src_x = 0;
		float src_y = 0;

		affine_project_device_kernel(&matrix, dst_x, dst_y, &src_x, &src_y);

		float c0 = padding_value, c1 = padding_value, c2 = padding_value;
		if (src_x < -1 || src_x >= src_w || src_y < -1 || src_y >= src_h) {
			// skip ...
		} else {
			int y_low = floorf(src_y); 
			int x_low = floorf(src_x); 
			int y_high = y_low + 1;
			int x_high = x_low + 1;

			uint8_t const_values[] = {  // channels == 3
				(uint8_t)padding_value, 
				(uint8_t)padding_value, 
				(uint8_t)padding_value }; 

			float ly = src_y - y_low;
			float lx = src_x - x_low;
			float hy = 1 - ly;
			float hx = 1 - lx;
			float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
			uint8_t* v1 = const_values;
			uint8_t* v2 = const_values;
			uint8_t* v3 = const_values;
			uint8_t* v4 = const_values;

            if (y_low >= 0) {
                if (x_low >= 0) {
					// src_volume: 单张像素数
					v1 = src + dy * src_volume + y_low * src_w * 3 + x_low * 3;
				}
                if (x_high < src_w) {
					v2 = src + dy * src_volume + y_low * src_w * 3 + x_high * 3;
				}
            }
            if (y_high < src_h) {
                if (x_low >= 0) {
					v3 = src + dy * src_volume + y_high * src_w * 3 + x_low * 3;
				}
                if (x_high < src_w) {
					v4 = src + dy * src_volume + y_high * src_w * 3 + x_high * 3;
				}
            }
            c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
			c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
			c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
		}  // end if-else
		float* pdst = dst + dy * dst_volume + dst_y * dst_w * 3 + dst_x * 3;
		pdst[0] = c0;
		pdst[1] = c1;
		pdst[2] = c2;
	}  // end if (dx < dst_area && dy < batch_size)

}

__device__ 
void affine_project_device_kernel(utils::AffineMat* matrix, int x, int y, float* proj_x, float* proj_y) {
	*proj_x = matrix->v0 * x + matrix->v1 * y + matrix->v2;
	*proj_y = matrix->v3 * x + matrix->v4 * y + matrix->v5;
}


/**
 * 单张图片预处理
 * @details 适用于 NHWC 格式（通道交错）
 */
__global__ 
void resize_device_kernel(uint8_t* src, int src_w, int src_h, float* dst, int dst_w, int dst_h, 
						float pad_value, utils::AffineMat matrix) {
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;

	if (dx >= dst_w || dy >= dst_h) {
		printf("resize_device_kernel cross the border");
		return;
	}
	int dst_x = dx;  // 列索引
	int dst_y = dy;  // 行索引

	float src_x = 0;
	float src_y = 0;

	affine_project_device_kernel(&matrix, dst_x, dst_y, &src_x, &src_y);

	float c0 = pad_value, c1 = pad_value, c2 = pad_value;
	if (src_x < -1 || src_x >= src_w || src_y < -1 || src_y >= src_h) {
		// skip ...
	} else {
		int y_low = floorf(src_y); 
		int x_low = floorf(src_x); 
		int y_high = y_low + 1;
		int x_high = x_low + 1;

		uint8_t const_values[] = {  // channels == 3
			(uint8_t)pad_value, 
			(uint8_t)pad_value, 
			(uint8_t)pad_value }; 

		float ly = src_y - y_low;
		float lx = src_x - x_low;
		float hy = 1 - ly;
		float hx = 1 - lx;
		float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
		uint8_t* v1 = const_values;
		uint8_t* v2 = const_values;
		uint8_t* v3 = const_values;
		uint8_t* v4 = const_values;

		if (y_low >= 0) {
			if (x_low >= 0) {
				// y_low lies in height
				v1 = src + y_low * src_w * 3 + x_low * 3;
			}
			if (x_high < src_w) {
				v2 = src + y_low * src_w * 3 + x_high * 3;
			}
		}
		if (y_high < src_h) {
			if (x_low >= 0) {
				v3 = src + y_high * src_w * 3 + x_low * 3;
			}
			if (x_high < src_w) {
				v4 = src + y_high * src_w * 3 + x_high * 3;
			}
		}
		c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
		c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
		c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
	}  // end if-else

	// float* pdst = dst + (dy * dst_w + dx) * 3;
    // pdst[0] = c0;
    // pdst[1] = c1;
    // pdst[2] = c2;

	int area = dst_w * dst_h;
	float *pdst_c0 = dst + dy * dst_w + dx;
	float *pdst_c1 = pdst_c0 + area;
	float *pdst_c2 = pdst_c1 + area;
	*pdst_c0 = c0;
	*pdst_c1 = c1;
	*pdst_c2 = c2;
}


/**
 * 交换 RB 通道，每线程处理一个像素位置，原地保存
 * 适用于 CHW 排列
 * @param order src 的通道排列顺序
 */
__global__ 
void swap_rb_channels_kernel_chw(float* src, int width, int height, utils::ChannelsArrange order) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;  // 行索引

    if (x >= width || y >= height) {
        return;
    }
    int image_size = width * height;
	int r_pos, g_pos, b_pos;
	if (order == utils::ChannelsArrange::RGB) {
		r_pos = y * width + x;
		g_pos = image_size + y * width + x;
		b_pos = 2 * image_size + y * width + x;
	} else if (order == utils::ChannelsArrange::BGR) {
		b_pos = y * width + x;                   // B 通道的位置（第 0 个平面）
		g_pos = image_size + y * width + x;      // G 通道的位置（第 1 个平面）
		r_pos = 2 * image_size + y * width + x;  // R 通道的位置（第 2 个平面）
	} else {
		// INFO("swap_channels unsupported [%d]", order);
		return;
	}
    float temp = src[b_pos];      // 保存 B
    src[b_pos] = src[r_pos];      // B <- R
    src[r_pos] = temp;            // R <- 原来的 B
}

/**
 * 适用于 RGB 或者 BGR 的 CHW 排列的图片
 * @param order src 的通道排列顺序
 */
__global__ 
void normalize_kernel_chw(float* src, int width, int height, utils::Norm norm, utils::ChannelsArrange order) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;  // 行索引

    int image_size = width * height;

	float *src_r, *src_g, *src_b;
    if (order == utils::ChannelsArrange::RGB) {
        src_r = src + y * width + x;
        src_g = src + image_size + y * width + x;
        src_b = src + 2 * image_size + y * width + x;
    } else if (order == utils::ChannelsArrange::BGR) {
        src_b = src + y * width + x;                   // B 通道
        src_g = src + image_size + y * width + x;      // G 通道
        src_r = src + 2 * image_size + y * width + x;  // R 通道
    } else {
		// INFO("normalize_kernel unsupported [%d]", order);
		return;
	}
    float r = *src_r;
    float g = *src_g;
    float b = *src_b;

    if (norm.type == utils::NormType::MeanStd) {
		r = (r * norm.alpha - norm.mean[0]) / norm.std[0];
        g = (g * norm.alpha - norm.mean[1]) / norm.std[1];
        b = (b * norm.alpha - norm.mean[2]) / norm.std[2];
    } else if (norm.type == utils::NormType::AlphaBeta) {
        r = r * norm.alpha + norm.beta;
        g = g * norm.alpha + norm.beta;
        b = b * norm.alpha + norm.beta;
    }
	// 按照输入排列的顺序保存输出
	float *dst_r, *dst_g, *dst_b;
	if (order == utils::ChannelsArrange::RGB) {
		dst_r = src + y * width + x;
		dst_g = src + image_size + y * width + x;
		dst_b = src + 2 * image_size + y * width + x;
	} else if (order == utils::ChannelsArrange::BGR) {
		dst_b = src + y * width + x;
		dst_g = src + image_size + y * width + x;
		dst_r = src + 2 * image_size + y * width + x;
	}
    *dst_r = r;
    *dst_g = g;
    *dst_b = b;
}
