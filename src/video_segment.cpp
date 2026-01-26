//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// A minimal Intel® Video Processing Library (Intel® VPL) decode, vpp and infer application,
/// using 2.x API with internal memory management,
/// showing zerocopy with remoteblob
///
/// @file

#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

#if defined(_WIN32) || defined(_WIN64)
#include <openvino/runtime/intel_gpu/ocl/dx.hpp>
#else
#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#endif
#include <openvino/runtime/intel_gpu/properties.hpp>

#include "./vpp.h" //NOLINT
#include <algorithm>
#include "./hw-device.h"
#include "./util.h"
#include "VideoSegment.h"
#define BITSTREAM_BUFFER_SIZE      2000000
#define SYNC_TIMEOUT               60000
#define MAJOR_API_VERSION_REQUIRED 2
#define MINOR_API_VERSION_REQUIRED 2


VideoSegmentor::VideoSegmentor()
{
}
VideoSegmentor::~VideoSegmentor()
{
}
void VideoSegmentor::PrintInputAndOutputsInfo(const ov::Model& network) {
    std::cout << "    Model name: " << network.get_friendly_name() << std::endl;

    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node> input : inputs) {
        std::cout << "    Inputs" << std::endl;

        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        std::cout << "        Input name: " << name << std::endl;

        const ov::element::Type type = input.get_element_type();
        std::cout << "        Input type: " << type << std::endl;

        const ov::Shape shape = input.get_shape();
        std::cout << "        Input shape: " << shape << std::endl;
    }

    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node> output : outputs) {
        std::cout << "    Outputs" << std::endl;

        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        std::cout << "        Output name: " << name << std::endl;

        const ov::element::Type type = output.get_element_type();
        std::cout << "        Output type: " << type << std::endl;

        const ov::Shape shape = output.get_shape();
        std::cout << "        Output shape: " << shape << std::endl;
    }

    std::cout << std::endl;
}
bool VideoSegmentor::Init(VideoSegParams Params)
{ 
    bool bIsSharedContextReady = false;

    try {

        //-- Get runtime version
        std::cout << ov::get_openvino_version() << std::endl;

        //-- Read a network model
        const std::string modelPath = TSTRING2STRING(Params.inmodelName);

        std::cout << "Loading network model files: " << modelPath << std::endl;
        std::shared_ptr<ov::Model> model = core.read_model(modelPath);
        PrintInputAndOutputsInfo(*model);

        VERIFY_THROW(model->inputs().size() == 1, "ERROR: sample supports topologies with 1 input only");
        VERIFY_THROW(model->outputs().size() == 1,
            "ERROR: sample supports topologies with 1 output only");

        auto input = model->input();
        auto output = model->output();

        m_strInputTensorName = input.get_any_name();
        m_strOutputTensorName = output.get_any_name();

        std::cout << "output Tensorname:" << m_strOutputTensorName << std::endl;
        std::cout << "input Tensorname:" << m_strInputTensorName << std::endl;
        ov::Shape inputShape = input.get_shape();
        ov::Layout inputLayout = ov::layout::get_layout(input);

        // 检查输入形状是否符合预期（至少有4个维度）
        int inputDimHeight =0;
        int inputDimWidth  =0;
        if (inputShape.size() == 4) {
            // 提取高度和宽度信息
            inputDimHeight = inputShape[2];
            inputDimWidth  = inputShape[3];

            std::cout << "Input height: " << inputDimHeight << std::endl;
            std::cout << "Input width: " << inputDimWidth << std::endl;
        }
        else {
            inputDimHeight = Params.inferHeight;
            inputDimWidth = Params.inferWidth;
            std::cerr << "Input shape does not have the expected 4 dimensions. fallback to recommed inference width & height" << std::endl;
        }

        //-- Configure preprocessing
        ov::preprocess::PrePostProcessor ppp(model);
        auto& inputInfo = ppp.input(m_strInputTensorName);

        // Set the input tensor
        //if (cliParams.bZeroCopy) {
        inputInfo.tensor()
            .set_element_type(ov::element::u8)
            .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, { "y", "uv" })
            .set_memory_type(ov::intel_gpu::memory_type::surface);
        //}
        //else
        //{
        //    inputInfo.tensor()
        //        .set_element_type(ov::element::u8)
        //        .set_color_format(ColorFormat::NV12_TWO_PLANES, { "y", "uv" });
        //}

        // Convert vpp output to BGR plannar
        inputInfo.preprocess()
            .convert_color(ov::preprocess::ColorFormat::BGR)
            .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR, inputDimWidth, inputDimHeight)
            .convert_element_type(ov::element::f32)
            .scale({ 255.0, 255.0, 255.0 })
            .mean({ 0.485, 0.456, 0.406 })
            .scale({ 0.229, 0.224, 0.225 });

        inputInfo.model().set_layout("NCHW");
        model = ppp.build();

#if defined(_WIN32) || defined(_WIN64)
        //-- Create inference request from shared context object
        auto sharedD3D11Context =
            ov::intel_gpu::ocl::D3DContext(core, Params.pD3D11Device);

        // Compile network within a shared context
        compiledModel = core.compile_model(model, sharedD3D11Context);
#else
        //-- Create inference request from shared context object
        auto sharedVAContext =
            ov::intel_gpu::ocl::VAContext(core, cliParams.lvaDisplay);

        // Compile network within a shared context
        compiledModel = core.compile_model(model, sharedVAContext);
#endif
        inferRequest = compiledModel.create_infer_request();
        //-- Start processing

    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n\n";

        return 1;
    }
    std::cout << "SceneDetection model is ready for inference " << std::endl;

    return true;
}
int VideoSegmentor::SceneDetect(void* pGenericTexture, int frameNo, bool bReset) {
    mfxStatus sts = MFX_ERR_NONE;


    //-- Infer from shared context and va surface
    auto context = compiledModel.get_context();
#if defined(_WIN32) || defined(_WIN64)
    auto& sharedContext =
        static_cast<ov::intel_gpu::ocl::D3DContext&>(context);
#else
    auto& sharedContext =
        static_cast<ov::intel_gpu::ocl::VAContext&>(context);
#endif


    /*InferFrame(sharedContext,
        pGenericTexture,
        inferRequest,
        m_strInputTensorName,
        m_strOutputTensorName,
        224,
        224);*/

   // frameIndex++;
    return 0;
}


using namespace ov::preprocess;
ov::InferRequest inferRequest;
ov::CompiledModel compiledModel;
ov::Core core;
std::vector<std::string> readImagenetLabels(const std::string& labelPath) {
    std::vector<std::string> labels;
    std::ifstream file(labelPath);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            labels.push_back(line);
        }
        file.close();
    }
    return labels;
}

// Softmax 函数
std::vector<float> softmax(const std::vector<float>& scores) {
    std::vector<float> probabilities(scores.size());
    float sum = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        probabilities[i] = std::exp(scores[i]);
        sum += probabilities[i];
    }
    for (size_t i = 0; i < probabilities.size(); ++i) {
        probabilities[i] /= sum;
    }
    return probabilities;
}

// 输出前 k 个预测结果
void printTopKResults(const std::vector<float>& probabilities, const std::vector<std::string>& labels, int k) {
    std::vector<std::pair<float, int>> sortedProbs;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        sortedProbs.emplace_back(probabilities[i], static_cast<int>(i));
    }
    std::sort(sortedProbs.begin(), sortedProbs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
        });

    std::cout << "前 " << k << " 个预测结果:" << std::endl;
    for (int i = 0; i < k; ++i) {
        int classIndex = sortedProbs[i].second;
        std::string className = labels[classIndex];
        float classProb = sortedProbs[i].first;
        std::cout << i + 1 << ". 类别: " << className << ", 概率: " << classProb * 100 << "%\n";
    }
}

//-- Initialize Runtime Core

void PrintInputAndOutputsInfo(const ov::Model& network) {
    std::cout << "    Model name: " << network.get_friendly_name() << std::endl;

    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node> input : inputs) {
        std::cout << "    Inputs" << std::endl;

        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        std::cout << "        Input name: " << name << std::endl;

        const ov::element::Type type = input.get_element_type();
        std::cout << "        Input type: " << type << std::endl;

        const ov::Shape shape = input.get_shape();
        std::cout << "        Input shape: " << shape << std::endl;
    }

    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node> output : outputs) {
        std::cout << "    Outputs" << std::endl;

        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        std::cout << "        Output name: " << name << std::endl;

        const ov::element::Type type = output.get_element_type();
        std::cout << "        Output type: " << type << std::endl;

        const ov::Shape shape = output.get_shape();
        std::cout << "        Output shape: " << shape << std::endl;
    }

    std::cout << std::endl;
}
bool CheckOpenVINOEnvironment(ov::Core& core) {
    try {
        // 版本
        std::cout << "OpenVINO Version: " << ov::get_openvino_version() << std::endl;
    //const char* pluginPath = 
    //    std::getenv("OV_PLUGIN_PATH");
    //std::cout << "OV_PLUGIN_PATH=" << (pluginPath ? pluginPath : "(unset)") << std::endl;

    const char* pathEnv = std::getenv("PATH");
    if (pathEnv)
        std::cout << "PATH contains length=" << strlen(pathEnv) << std::endl;
        // 可用设备
        auto devices = core.get_available_devices();
        std::cout << "Available devices (" << devices.size() << "): ";
        for (auto& d : devices) std::cout << d << " ";
        std::cout << std::endl;
        if (devices.empty()) {
            std::cerr << "WARNING: No devices detected." << std::endl;
        }

        // GPU 插件检查（存在则输出属性）
        if (std::find(devices.begin(), devices.end(), "GPU") != devices.end()) {
            try {
                auto opt = core.get_property("GPU", ov::device::properties);
                std::cout << "GPU device properties count: " << opt.size() << std::endl;
            } catch (const std::exception& ex) {
                std::cerr << "GPU properties query failed: " << ex.what() << std::endl;
            }
        }

//        // 前端列表（若 frontend API 可用）
//        try {
//#if defined(OPENVINO_VERSION_MAJOR) // 假定已定义版本宏
//            // 新版前端 (若你的版本支持)
//            ov::frontend::FrontEndManager fem;
//            auto fets = ov::frontend::FrontEndManager().get_available_front_ends();
//            std::cout << "Available frontends: ";
//            for (auto& f : fets) std::cout << f << " ";
//            std::cout << std::endl;
//#endif
//        } catch (...) {
//            std::cout << "Frontend enumeration not supported in this build." << std::endl;
//        }

        // 简单功能测试：创建空张量
        ov::Tensor t{ov::element::f32, {1, 3, 2, 2}};
        if (t.get_size() != 12) {
            std::cerr << "Tensor sanity check failed." << std::endl;
            return false;
        }

        std::cout << "OpenVINO environment check passed." << std::endl;
        return true;
    } catch (const ov::Exception& ex) {
        std::cerr << "OpenVINO exception: " << ex.what() << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Std exception: " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception during environment check." << std::endl;
    }
    return false;
}
#if defined(_WIN32) || defined(_WIN64)
mfxStatus InferFrame(ov::intel_gpu::ocl::D3DContext context,
#else
mfxStatus InferFrame(ov::intel_gpu::ocl::VAContext context,
#endif
    void* surface,
    ov::InferRequest inferRequest,
    std::string inputName,
    std::string outputName,
    mfxU16 oriWidth,
    mfxU16 oriHeight);

std::string inputTensorName ;
std::string outputTensorName;
int InitVideoSegment(VideoSegParams cliParams) {
    //-- Params for decode/vpp session
    mfxStatus sts = MFX_ERR_NONE;
    FILE* source = NULL;

    //--> variables used only in legacy gen mode
    int accel_fd = 0;
    void* accelHandle = NULL;
    mfxFrameAllocRequest decRequest = {};
    mfxFrameAllocRequest vppRequest[2] = {};
    mfxU16 nSurfNumDecVPP = 0;
    mfxU16 nSurfNumVPPOut = 0;
    mfxFrameSurface1* pmfxDecSurfPool = NULL;
    mfxU8* decOutBuf = NULL;
    mfxFrameSurface1* pmfxVPPSurfPool = NULL;
    mfxU8* vppOutBuf = NULL;
    int nIndex = -1;
    int nIndex2 = -1;
    //<--

    mfxConfig cfg[4] = {};
    mfxVariant cfgVal[4] = {};
    bool bIsSharedContextReady = false;

    mfxU16 oriImgWidth, oriImgHeight;
    mfxU16 inputDimWidth, inputDimHeight;
    mfxU16 vppInImgWidth, vppInImgHeight;
    mfxU16 vppOutImgWidth, vppOutImgHeight;

    try {

        //-- Get runtime version
        std::cout << ov::get_openvino_version() << std::endl;

        //-- Initialize Runtime Core
        ov::Core core;

        if(!CheckOpenVINOEnvironment(core)) { std::cerr<<"Env check failed\n"; }


        //-- Read a network model
        const std::string modelPath = /*TSTRING2STRING(cliParams.inmodelName)*/ "D:\\models\\resnet50 - scene\\resnet50_scene.xml";
        std::cout << "Loading network model files: " << modelPath << std::endl;
        // D:\models\gemma3-4b-it-ov *D:\\models\\resnet-50-ov-int4*/
        std::shared_ptr<ov::Model> model = core.read_model("D:\\models\\resnet50-scene\\resnet50_scene.xml");
        PrintInputAndOutputsInfo(*model);

        VERIFY_THROW(model->inputs().size() == 1, "ERROR: sample supports topologies with 1 input only");
        VERIFY_THROW(model->outputs().size() == 1,
            "ERROR: sample supports topologies with 1 output only");

        auto input = model->input();
        auto output = model->output();

        inputTensorName = input.get_any_name();
        outputTensorName = output.get_any_name();

        // Check whether network model is for object detection
        //VERIFY_THROW(outputTensorName.compare("detection_out") == 0,
        //    "ERROR: must use object detection network model (ex: mobilenet-ssd)");

        std::cout << "output Tensorname:"<<outputTensorName<< std::endl;
        std::cout << "input Tensorname:" << inputTensorName << std::endl;
        ov::Shape inputShape = input.get_shape();
        ov::Layout inputLayout = ov::layout::get_layout(input);

        // 检查输入形状是否符合预期（至少有4个维度）
        if (inputShape.size() == 4) {
            // 提取高度和宽度信息
            inputDimHeight = inputShape[2];
            inputDimWidth = inputShape[3];

            std::cout << "Input height: " << inputDimHeight << std::endl;
            std::cout << "Input width: " << inputDimWidth << std::endl;
        }
        else {
            std::cerr << "Input shape does not have the expected 4 dimensions." << std::endl;
        }

        //-- Configure preprocessing
        PrePostProcessor ppp(model);
        InputInfo& inputInfo = ppp.input(inputTensorName);

        // Set the input tensor
        //if (cliParams.bZeroCopy) {
            inputInfo.tensor()
                .set_element_type(ov::element::u8)
                .set_color_format(ColorFormat::NV12_TWO_PLANES, { "y", "uv" })
                .set_memory_type(ov::intel_gpu::memory_type::surface);
        //}
        //else
        //{
        //    inputInfo.tensor()
        //        .set_element_type(ov::element::u8)
        //        .set_color_format(ColorFormat::NV12_TWO_PLANES, { "y", "uv" });
        //}

        // Convert vpp output to BGR plannar
        

        inputInfo.preprocess()
            .convert_color(ov::preprocess::ColorFormat::BGR)
            .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR, 224, 224)
            .convert_element_type(ov::element::f32)
            .scale({ 255.0, 255.0, 255.0 })
            .mean({ 0.485, 0.456, 0.406 })
            .scale({ 0.229, 0.224, 0.225 });


        //inputInfo.preprocess().convert_color(ov::preprocess::ColorFormat::BGR);

        inputInfo.model().set_layout("NCHW");

        model = ppp.build();

#if defined(_WIN32) || defined(_WIN64)
        //-- Create inference request from shared context object
        auto sharedD3D11Context =
            ov::intel_gpu::ocl::D3DContext(core, cliParams.pD3D11Device);

        // Compile network within a shared context
        compiledModel = core.compile_model(model, sharedD3D11Context);
#else
        //-- Create inference request from shared context object
        auto sharedVAContext =
            ov::intel_gpu::ocl::VAContext(core, cliParams.lvaDisplay);

        // Compile network within a shared context
        compiledModel = core.compile_model(model, sharedVAContext);
#endif
        inferRequest = compiledModel.create_infer_request();
        //-- Start processing

    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n\n";

        return 1;
    }
    std::cout << "SceneDetection model is ready for inference " << std::endl;
    return 0;
}
//bool LoadAndReportModel(const std::string& xmlPath,
//                        ov::Core& core,
//                        std::shared_ptr<ov::Model>& outModel) {
//    namespace fs = std::filesystem;
//    try {
//        // 1. 路径与文件检查
//        if (!fs::exists(xmlPath)) {
//            std::cerr << "模型XML不存在: " << xmlPath << std::endl;
//            return false;
//        }
//        fs::path binPath = fs::path(xmlPath).replace_extension(".bin");
//        if (!fs::exists(binPath)) {
//            std::cerr << "对应BIN文件缺失: " << binPath.string() << std::endl;
//            return false;
//        }
//
//        // 2. 调用 OpenVINO 读模型
//        outModel = core.read_model(xmlPath);
//
//        if (!outModel) {
//            std::cerr << "read_model 返回空指针" << std::endl;
//            return false;
//        }
//
//        // 3. 打印基本信息
//        std::cout << "==== 模型加载成功 ====" << std::endl;
//        std::cout << "模型友好名称: " << outModel->get_friendly_name() << std::endl;
//        std::cout << "XML: " << xmlPath << "\nBIN: " << binPath.string() << std::endl;
//
//        // 输入信息
//        auto inputs = outModel->inputs();
//        std::cout << "输入数: " << inputs.size() << std::endl;
//        for (size_t i = 0; i < inputs.size(); ++i) {
//            auto in = inputs[i];
//            std::string name = in.get_names().empty() ? "(unnamed)" : in.get_any_name();
//            std::cout << "  [Input " << i << "] name=" << name
//                      << " type=" << in.get_element_type()
//                      << " shape=" << in.get_shape() << std::endl;
//            // 尝试获取布局
//            try {
//                auto layout = ov::layout::get_layout(in);
//                std::cout << "    layout=" << layout.to_string() << std::endl;
//            } catch (...) {
//                std::cout << "    layout=未定义" << std::endl;
//            }
//        }
//
//        // 输出信息
//        auto outputs = outModel->outputs();
//        std::cout << "输出数: " << outputs.size() << std::endl;
//        for (size_t i = 0; i < outputs.size(); ++i) {
//            auto out = outputs[i];
//            std::string name = out.get_names().empty() ? "(unnamed)" : out.get_any_name();
//            std::cout << "  [Output " << i << "] name=" << name
//                      << " type=" << out.get_element_type()
//                      << " shape=" << out.get_shape() << std::endl;
//        }
//
//        std::cout << "======================" << std::endl;
//        return true;
//    } catch (const ov::Exception& ex) {
//        std::cerr << "OpenVINO 异常: " << ex.what() << std::endl;
//    } catch (const std::exception& ex) {
//        std::cerr << "标准异常: " << ex.what() << std::endl;
//    } catch (...) {
//        std::cerr << "未知异常" << std::endl;
//    }
//    return false;
//}
//

#include <cmath>

static unsigned int frameIndex = 0;
std::vector<int> sceneChanges = { 0 };  // 第一帧是场景的起始
std::vector<float> previousFeature;
unsigned int queueSize = 0;

std::deque<std::vector<float>> featureWindow;  // 滑动窗口，存储前面 4 帧的特征向量
const size_t WINDOW_SIZE = 10;
// 计算余弦相似度
double cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    normA = std::sqrt(normA);
    normB = std::sqrt(normB);

    if (normA == 0 || normB == 0) {
        return 0.0;
    }

    return dotProduct / (normA * normB);
}



// 根据语义相似性检测场景变化
int detectScenesBySemanticSimilarity(const std::vector<float>& feature, double threshold = 0.7) {
    if (featureWindow.size() < WINDOW_SIZE) {
        // 前 WINDOW_SIZE 帧，先填充窗口
        featureWindow.push_back(feature);
    }
    else {
        bool isSceneChange = false;
        // 计算当前帧与窗口内每一帧的相似度
        int diff = 0;
        for (const auto& prevFeature : featureWindow) {
            double similarity = cosineSimilarity(prevFeature, feature);
            if (similarity >= threshold) {
                break;
            }
            else {
                diff++;
            }
        }

        if (diff == WINDOW_SIZE) {
            isSceneChange = true;
        }

        if (isSceneChange) {
            featureWindow.clear();
            sceneChanges.push_back(static_cast<int>(frameIndex));
        }
        else {
            featureWindow.pop_front();
            featureWindow.push_back(feature);
        }
    }
    std::cout << "Scene change at frame # " << frameIndex <<std::endl;
    for (int frameNum : sceneChanges) {
        std::cout << frameNum << " ";
    }
    std::cout << std::endl;
    //frameIndex++;
    return 0;
}

// 根据语义相似性检测场景变化
int detectScenesBySemanticSimilarity2( std::vector<float>& feature, double threshold = 0.7) {

    if (frameIndex == 0) {
       // skip scene detect, assume it is the first scene
        previousFeature.resize(feature.size());
        for (size_t i = 0; i < feature.size(); ++i) {
            previousFeature[i] = feature[i];
        }
    }
    else {
       
        double similarity = cosineSimilarity(previousFeature, feature);
        if (similarity < threshold) {  // 如果相似度低于阈值，认为场景发生变化
            sceneChanges.push_back(static_cast<int>(frameIndex));
        }
        for (size_t i = 0; i < feature.size(); ++i) {
            previousFeature[i] = feature[i];
        }
        
    }

    std::cout << "场景变化的帧号: ";
    for (int frameNum : sceneChanges) {
        std::cout << frameNum << " ";
    }
    std::cout << std::endl;
    return 0;
}

int SceneDetect(void *pGenericTexture) {

    mfxStatus sts = MFX_ERR_NONE;
                        
    //-- Infer from shared context and va surface
    auto context = compiledModel.get_context();
    if (!context) {
        std::cerr << "Model not initialized\n";
        return -1;
    }
#if defined(_WIN32) || defined(_WIN64)
    auto& sharedContext =
        static_cast<ov::intel_gpu::ocl::D3DContext&>(context);
#else
    auto& sharedContext =
        static_cast<ov::intel_gpu::ocl::VAContext&>(context);
#endif
    InferFrame(sharedContext,
        pGenericTexture,
        inferRequest,
        inputTensorName,
        outputTensorName,
        224,
        224);

    frameIndex++;
    return 0;
}

#if defined(_WIN32) || defined(_WIN64)
mfxStatus InferFrame(ov::intel_gpu::ocl::D3DContext context,
#else
mfxStatus InferFrame(ov::intel_gpu::ocl::VAContext context,
#endif
    void* pGenericSurface,
    ov::InferRequest inferRequest,
    std::string inputName,
    std::string outputName,
    mfxU16 oriWidth,
    mfxU16 oriHeight) {
    mfxStatus sts = MFX_ERR_NONE;
#if defined(_WIN32) || defined(_WIN64)
    ID3D11Texture2D* pD3D11Texture;
#else
    VASurfaceID lvaSurfaceID;
#endif
    //mfxHDL lresource;
    //mfxResourceType lresourceType;

  
   // std::cout << "Result: " << std::endl;

#if defined(_WIN32) || defined(_WIN64)
    pD3D11Texture = (ID3D11Texture2D*)pGenericSurface;
    // Wrap VPP output into remoteblobs and set it as inference input tensor
    auto nv12Tensor =
        context.create_tensor_nv12(224, 224, pD3D11Texture);
#else
    lvaSurfaceID = *(VASurfaceID*)lresource;
    // Wrap VPP output into remoteblobs and set it as inference input tensor
    auto nv12Tensor =
        context.create_tensor_nv12(surface->Info.CropH, surface->Info.CropW, lvaSurfaceID);
#endif

    inferRequest.set_input_tensor(0, nv12Tensor.first);
    inferRequest.set_input_tensor(1, nv12Tensor.second);

    //-- Infers specified input(s) in synchronous mode
    inferRequest.infer();

    /*auto outputTensor = inferRequest.get_tensor(outputName);

    // 获取输出张量
    const float* outputData = outputTensor.data<const float>();
    std::vector<float> scores(outputTensor.get_size());
    std::cout << "tensor size:"<< outputTensor.get_size()<<std::endl;
    */

    // 获取输出张量
    ov::Tensor outputTensor = inferRequest.get_output_tensor();
    const float* outputData = outputTensor.data<const float>();
    std::vector<float> feature(outputTensor.get_size());
    for (size_t i = 0; i < outputTensor.get_size(); ++i) {
        feature[i] = outputData[i];
    }
    detectScenesBySemanticSimilarity(feature, 0.7);
 
    std::cout << std::endl;
    return sts;
}


#if 0
int main()
{
    VideoSegParams params;
    params.bZeroCopy = TRUE;
    params.pD3D11Device = NULL;
    params.inmodelName = (char*)("D:\\models\\resnet50.xml");
    InitVideoSegment(params);
    return 0;
}
#endif