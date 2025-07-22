#pragma once

#include <functional>
#include <string>
#include <vector>

#include "common/cuda_tools.h"
#include "infer/trt_infer.h"

namespace TRT {

enum class ModelSourceType : int { ONNX, ONNXDATA };

class ModelSource {
public:
    ModelSource() = default;
    ModelSource(const std::string& onnxmodel);
    ModelSource(const char* onnxmodel);
    ModelSourceType type() const;
    std::string     onnxmodel() const;
    std::string     descript() const;
    const void*     onnx_data() const;
    size_t          onnx_data_size() const;

    static ModelSource onnx(const std::string& file) {
        ModelSource output;
        output.onnxmodel_ = file;
        output.type_ = ModelSourceType::ONNX;
        return output;
    }
    static ModelSource onnx_data(const void* ptr, size_t size) {
        ModelSource output;
        output.onnx_data_ = ptr;
        output.onnx_data_size_ = size;
        output.type_ = ModelSourceType::ONNXDATA;
        return output;
    }

private:
    std::string     onnxmodel_;
    const void*     onnx_data_ = nullptr;
    size_t          onnx_data_size_ = 0;
    ModelSourceType type_;
};

enum class Mode : int { FP32, FP16, INT8 };

const char* mode_string(Mode type);

class InputDims {
public:
    InputDims() = default;
    InputDims(const std::initializer_list<int>& dims);  // 当为-1时，保留导入时的网络结构尺寸
    InputDims(const std::vector<int>& dims);
    const std::vector<int>& dims() const;

private:
    std::vector<int> dims_;
};

enum class CompileOutputType : int { File, Memory };

class CompileOutput {
public:
    CompileOutput(CompileOutputType type = CompileOutputType::File);
    CompileOutput(const std::string& file);
    CompileOutput(const char* file);
    void set_data(const std::vector<uint8_t>& data);
    void set_data(std::vector<uint8_t>&& data);

    const std::vector<uint8_t>& data() const { return data_; };
    CompileOutputType           type() const { return type_; }
    std::string                 file() const { return file_; }

private:
    CompileOutputType    type_ = CompileOutputType::File;
    std::vector<uint8_t> data_;
    std::string          file_;
};

bool compile(Mode mode, uint32_t maxBatchSize, const ModelSource& source, const CompileOutput& saveto,
             const size_t maxWorkspaceSize = 2ULL << 30);

}  // namespace TRT
