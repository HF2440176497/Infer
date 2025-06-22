#pragma once

#include <NvInfer.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <logger.h>
#include <parserOnnxConfig.h>
#include <thrust/sort.h>

#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <algorithm>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <stdexcept>

#include <vector>
#include <string>

#include <stdio.h>