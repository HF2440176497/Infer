#pragma once

// ==================== TensorRT Includes ====================
#include <NvInfer.h>

// ==================== CUDA Includes ====================
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <cuda_fp16.h>

// ==================== Thrust Includes ====================
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// ==================== OpenCV Includes ====================
#include <opencv2/opencv.hpp>

// ==================== C Standard Library Includes ====================
#include <cstdlib>
#include <cstdint>
#include <cstdarg>
#include <stdio.h>

// ==================== C++ Standard Library Includes ====================
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <map>
#include <tuple>
#include <future>

// ==================== Other Includes ====================
// 