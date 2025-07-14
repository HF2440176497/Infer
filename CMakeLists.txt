
cmake_minimum_required(VERSION 3.26)

project(main LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 显卡计算能力 4050 -> 8.9
set(CMAKE_CUDA_ARCHITECTURES "89") 

# 设置
set(OpenCV_DIR   "/usr/local/lib/cmake/opencv4")
set(CUDA_TOOLKIT_ROOT_DIR     "/usr/local/cuda")

set(TensorRT_ROOT "/usr/src/tensorrt")
set(TENSORRT_SAMPLES_DIR ${TensorRT_ROOT}/samples)

find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
find_package(CUDAToolkit REQUIRED)

set(main_srcs 
   ${PROJECT_SOURCE_DIR}/main.cpp)

file(GLOB_RECURSE utils_srcs 
    ${PROJECT_SOURCE_DIR}/utils/*.cpp
    ${PROJECT_SOURCE_DIR}/utils/*.cu)

file(GLOB_RECURSE yolo_srcs 
    ${PROJECT_SOURCE_DIR}/application/*.cpp
    ${PROJECT_SOURCE_DIR}/application/*.cu)

file(GLOB_RECURSE trt_srcs
    ${PROJECT_SOURCE_DIR}/TensorRT/*.cpp
    ${PROJECT_SOURCE_DIR}/TensorRT/*.cu)

add_executable(main
    ${utils_srcs}
    ${yolo_srcs}
    ${trt_srcs}
    ${main_srcs}
    # ${TensorRT_ROOT}/samples/common/logger.cpp 
    # ${TensorRT_ROOT}/samples/common/sampleOptions.cpp
)

# 查看 OpenCv 头文件目录
# message(STATUS "OpenCV include found at: ${OpenCV_INCLUDE_DIRS}")

target_include_directories(main PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/application
    ${PROJECT_SOURCE_DIR}/utils
    ${PROJECT_SOURCE_DIR}/TensorRT
    ${PROJECT_SOURCE_DIR}/TensorRT/builder
    ${PROJECT_SOURCE_DIR}/TensorRT/common
    ${PROJECT_SOURCE_DIR}/TensorRT/infer
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/include
)

# 链接库目录
target_link_directories(main PRIVATE
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

# 链接库
target_link_libraries(main 
    CUDA::cudart 
    CUDA::cublas
    nvinfer
    nvinfer_plugin
    nvonnxparser
)

target_link_libraries(main 
    pthread
)

target_link_libraries(main 
    ${OpenCV_LIBS}
)
