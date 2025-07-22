

#include "builder/trt_builder.hpp"
#include "common/ilog.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

class Logger : public ILogger {
public:
    virtual void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kINTERNAL_ERROR) {
            INFO("NVInfer INTERNAL_ERROR: %s", msg);
            abort();
        } else if (severity == Severity::kERROR) {
            INFO("NVInfer: %s", msg);
        } else if (severity == Severity::kWARNING) {
            INFO("NVInfer: %s", msg);
        } else if (severity == Severity::kINFO) {
            INFO("NVInfer: %s", msg);
        } else {
            INFO("%s", msg);
        }
    }
};
static Logger gLogger;

namespace TRT {

static std::string join_dims(const std::vector<int>& dims) {
    std::stringstream output;
    char              buf[64];
    const char*       fmts[] = {"%d", " x %d"};
    for (int i = 0; i < dims.size(); ++i) {
        snprintf(buf, sizeof(buf), fmts[i != 0], dims[i]);
        output << buf;
    }
    return output.str();
}

static std::string format(const char* fmt, ...) {
    va_list vl;
    va_start(vl, fmt);
    char buffer[10000];
    vsprintf(buffer, fmt, vl);
    return buffer;
}

template <typename _T>
static void destroy_nvidia_pointer(_T* ptr) {
    if (ptr) delete ptr;
}

const char* mode_string(Mode type) {
    switch (type) {
    case Mode::FP32:
        return "FP32";
    case Mode::FP16:
        return "FP16";
    case Mode::INT8:
        return "INT8";
    default:
        return "UnknowTRTMode";
    }
}

const std::vector<int>& InputDims::dims() const { return dims_; }

InputDims::InputDims(const std::initializer_list<int>& dims) : dims_(dims) {}

InputDims::InputDims(const std::vector<int>& dims) : dims_(dims) {}

ModelSource::ModelSource(const char* onnxmodel) {
    this->type_ = ModelSourceType::ONNX;
    this->onnxmodel_ = onnxmodel;
}

ModelSource::ModelSource(const std::string& onnxmodel) {
    this->type_ = ModelSourceType::ONNX;
    this->onnxmodel_ = onnxmodel;
}

const void* ModelSource::onnx_data() const { return this->onnx_data_; }

size_t ModelSource::onnx_data_size() const { return this->onnx_data_size_; }

std::string     ModelSource::onnxmodel() const { return this->onnxmodel_; }
ModelSourceType ModelSource::type() const { return this->type_; }
std::string     ModelSource::descript() const {
    if (this->type_ == ModelSourceType::ONNX)
        return format("ONNX Model '%s'", onnxmodel_.c_str());
    else if (this->type_ == ModelSourceType::ONNXDATA)
        return format("ONNXDATA Data: '%p', Size: '%lld'", onnx_data_, onnx_data_size_);
    else
	return "Not support source type";
}

CompileOutput::CompileOutput(CompileOutputType type) : type_(type) {}
CompileOutput::CompileOutput(const std::string& file) : type_(CompileOutputType::File), file_(file) {}
CompileOutput::CompileOutput(const char* file) : type_(CompileOutputType::File), file_(file) {}
void CompileOutput::set_data(const std::vector<uint8_t>& data) { data_ = data; }

void CompileOutput::set_data(std::vector<uint8_t>&& data) { data_ = std::move(data); }

/**
 * 采用动态批次
 * maxSize in bytes, 1ULL << 30 represents 1 GB
 */
bool compile(Mode mode, uint32_t maxBatchSize, const ModelSource& source, const CompileOutput& saveto,
             const size_t maxWorkspaceSize) {
    INFO("Compile %s %s.", mode_string(mode), source.descript().c_str());
    std::shared_ptr<IBuilder> builder(createInferBuilder(gLogger), destroy_nvidia_pointer<IBuilder>);
    if (builder == nullptr) {
        INFO("Can not create builder.");
        return false;
    }
    std::shared_ptr<INetworkDefinition> network(builder->createNetworkV2(0), destroy_nvidia_pointer<INetworkDefinition>);

    if (network == nullptr) {
        INFO("Can not create network.");
        return false;
    }
    std::shared_ptr<IBuilderConfig> config(builder->createBuilderConfig(), destroy_nvidia_pointer<IBuilderConfig>);
    if (config == nullptr) {
        INFO("Can not create config.");
        return false;
    }
    std::shared_ptr<nvonnxparser::IParser> onnxParser(nvonnxparser::createParser(*network, gLogger),
                                                      destroy_nvidia_pointer<nvonnxparser::IParser>);
    if (onnxParser == nullptr) {
        INFO("Can not create parser.");
        return false;
    }
    if (source.type() == ModelSourceType::ONNX) {
        if (!onnxParser->parseFromFile(source.onnxmodel().c_str(),
                                       static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
            INFO("Can not parse OnnX file: %s", source.onnxmodel().c_str());
            return false;
        }
    } else {
        // Parse a serialized ONNX model
        if (!onnxParser->parse(source.onnx_data(), source.onnx_data_size())) {
            INFO("Can not parse OnnX file: %s", source.onnxmodel().c_str());
            return false;
        }
    }
    auto inputDims = network->getInput(0)->getDimensions();
    INFO("Input shape is %s", join_dims(std::vector<int>(inputDims.d, inputDims.d + inputDims.nbDims)).c_str());
    INFO("Set max batch size = %d", maxBatchSize);
    INFO("Set max workspace size = %.2f MB", maxWorkspaceSize / 1024.0f / 1024.0f);  // bytes to MB
    INFO("Base device: %s", CUDATools::device_description().c_str());

    int net_num_input = network->getNbInputs();
    INFO("Network has %d inputs:", net_num_input);
    std::vector<std::string> input_names(net_num_input);
    for (int i = 0; i < net_num_input; ++i) {
        auto tensor = network->getInput(i);
        auto dims = tensor->getDimensions();
        auto dims_str = join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
        INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());
        input_names[i] = tensor->getName();
    }

    int net_num_output = network->getNbOutputs();
    INFO("Network has %d outputs:", net_num_output);
    for (int i = 0; i < net_num_output; ++i) {
        auto tensor = network->getOutput(i);
        auto dims = tensor->getDimensions();
        auto dims_str = join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
        INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());
    }

    network->getInput(0)->setAllowedFormats(1U << static_cast<int32_t>(TensorFormat::kLINEAR));
    network->getOutput(0)->setAllowedFormats(1U << static_cast<int32_t>(TensorFormat::kLINEAR));

    network->getInput(0)->setType(nvinfer1::DataType::kFLOAT);
    network->getOutput(0)->setType(nvinfer1::DataType::kFLOAT);
    
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, maxBatchSize);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    auto profile = builder->createOptimizationProfile();
    for (int i = 0; i < net_num_input; ++i) {
        auto input = network->getInput(i);
        auto input_dims = input->getDimensions();  // N C H W
        input_dims.d[0] = 2;
        input_dims.d[2] = 640;
        input_dims.d[3] = 640;
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);

        input_dims.d[0] = maxBatchSize;
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    }
    config->addOptimizationProfile(profile);

    INFO("Building engine...");
    auto                         time_start = utils::timestamp_ms();
    std::shared_ptr<ICudaEngine> engine(builder->buildEngineWithConfig(*network, *config),
                                        destroy_nvidia_pointer<ICudaEngine>);
    if (engine == nullptr) {
        INFO("engine is nullptr");
        return false;
    }

    INFO("Build done %lld ms !", utils::timestamp_ms() - time_start);

    // serialize the engine, then close everything down
    std::shared_ptr<IHostMemory> seridata(engine->serialize(), destroy_nvidia_pointer<IHostMemory>);
    if (saveto.type() == CompileOutputType::File) {
        return iLog::save_file(saveto.file(), seridata->data(), seridata->size());
    } else {
	const_cast<CompileOutput&>(saveto).set_data(
            std::vector<uint8_t>((uint8_t*)seridata->data(), (uint8_t*)seridata->data() + seridata->size()));
        return true;
    }
}

}  // namespace TRT
