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

#include <openvino/openvino.hpp>

#ifdef ZEROCOPY
#if defined(_WIN32) || defined(_WIN64)
#include <openvino/runtime/intel_gpu/ocl/dx.hpp>
#else
#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#endif
#include <openvino/runtime/intel_gpu/properties.hpp>
#endif

#include "util.hpp"

#define BITSTREAM_BUFFER_SIZE      2000000
#define SYNC_TIMEOUT               60000
#define MAJOR_API_VERSION_REQUIRED 2
#define MINOR_API_VERSION_REQUIRED 2

using namespace ov::preprocess;

void Usage(void) {
    printf("\n");
    printf("   Usage    :    vpl-infer \n\n");
    printf("     -i          input file name (HEVC elementary stream)\n");
    printf("     -m          input model name (object detection)\n");
#ifdef ZEROCOPY
    printf("     -zerocopy   process without creating an additional copy of the data\n");
#endif
    printf("     -legacy     run sample in legacy gen (ex: gen 9.x - SKL, KBL, CFL, etc)\n\n");
    printf("   Example  :    vpl-infer -i in.h265 -m mobilenet-ssd.xml\n\n");
    return;
}

mfxSession CreateVPLSession(mfxLoader* loader, Params* cli);

#ifdef ZEROCOPY
#if defined(_WIN32) || defined(_WIN64)
mfxStatus InferFrame(ov::intel_gpu::ocl::D3DContext context,
#else
mfxStatus InferFrame(ov::intel_gpu::ocl::VAContext context,
#endif
    mfxFrameSurface1* surface,
    ov::InferRequest inferRequest,
    std::string inputName,
    std::string outputName,
    mfxU16 oriWidth,
    mfxU16 oriHeight);
#endif

void InferFrame(mfxFrameSurface1* surface,
    ov::InferRequest inferRequest,
    std::string inputName,
    std::string outputName,
    mfxU16 oriWidth,
    mfxU16 oriHeight);

#ifdef ZEROCOPY
#if defined(_WIN32) || defined(_WIN64)
mfxStatus InferFrame(ov::intel_gpu::ocl::D3DContext context,
#else
mfxStatus InferFrame(ov::intel_gpu::ocl::VAContext context,
#endif
    mfxFrameSurface1* surface,
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
    mfxHDL lresource;
    mfxResourceType lresourceType;

    sts = surface->FrameInterface->GetNativeHandle(surface, &lresource, &lresourceType);

    if (sts != MFX_ERR_NONE)
        return sts;

    std::cout << "Result: " << std::endl;

#if defined(_WIN32) || defined(_WIN64)
    pD3D11Texture = (ID3D11Texture2D*)lresource;
    // Wrap VPP output into remoteblobs and set it as inference input tensor
    auto nv12Tensor =
        context.create_tensor_nv12(surface->Info.CropH, surface->Info.CropW, pD3D11Texture);
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

    auto outputTensor = inferRequest.get_tensor(outputName);

    // Display class id, bounding box rect info, and confidence score of detected object
    size_t lastDim = outputTensor.get_shape().back();

    if (lastDim == 7) {
        float* data = (float*)outputTensor.data();
        for (size_t i = 0; i < outputTensor.get_size() / lastDim; i++) {
            int imageId = static_cast<int>(data[i * lastDim + 0]);
            int classId = static_cast<int>(data[i * lastDim + 1]);
            float confidence = data[i * lastDim + 2];
            auto x_min = static_cast<int>(data[i * lastDim + 3] * oriWidth);
            auto y_min = static_cast<int>(data[i * lastDim + 4] * oriHeight);
            auto x_max = static_cast<int>(data[i * lastDim + 5] * oriWidth);
            auto y_max = static_cast<int>(data[i * lastDim + 6] * oriHeight);
            if (imageId < 0)
                break;
            if (confidence < 0.5) {
                continue;
            }

            printf("    Class ID (%d),  BBox (%4d, %4d, %4d, %4d),  Confidence (%5.3f)\n",
                classId,
                x_min,
                y_min,
                x_max,
                y_max,
                confidence);
        }
    }

    std::cout << std::endl;
    return sts;
}
#endif

void InferFrame(mfxFrameSurface1* surface,
    ov::InferRequest inferRequest,
    std::string inputName,
    std::string outputName,
    mfxU16 oriWidth,
    mfxU16 oriHeight) {
    mfxFrameInfo* info = &surface->Info;
    mfxFrameData* data = &surface->Data;

    size_t w = info->CropW;
    size_t h = info->CropH;
    size_t p = data->Pitch;

    std::cout << "Result: " << std::endl;

    // Prepare input tensor with copying mfxFrameSurface data
    ov::Tensor inputTensorY{ ov::element::u8, { 1, h, w, 1 } };
    unsigned char* pDataY = (unsigned char*)inputTensorY.data();

    for (int i = 0; i < h; i++)
        memcpy(pDataY + i * w, (unsigned char*)(data->Y + (i * p)), w);

    if (info->FourCC == MFX_FOURCC_I420) {
        p = p / 2;
        h = h / 2;
        w = w / 2;

        ov::Tensor inputTensorU{ ov::element::u8, { 1, h, w, 1 } };
        ov::Tensor inputTensorV{ ov::element::u8, { 1, h, w, 1 } };

        // U
        unsigned char* pDataU = (unsigned char*)inputTensorU.data();

        for (int i = 0; i < h; i++)
            memcpy(pDataU + i * w, (unsigned char*)(data->U + (i * p)), w);

        // V
        unsigned char* pDataV = (unsigned char*)inputTensorV.data();

        for (int i = 0; i < h; i++)
            memcpy(pDataV + i * w, (unsigned char*)(data->V + (i * p)), w);

        inferRequest.set_input_tensor(0, inputTensorY);
        inferRequest.set_input_tensor(1, inputTensorU);
        inferRequest.set_input_tensor(2, inputTensorV);
    }
    else {
        // UV
        ov::Tensor inputTensorUV{ ov::element::u8, { 1, h / 2, w / 2, 2 } };
        unsigned char* pDataUV = (unsigned char*)inputTensorUV.data();

        h = h / 2;
        for (int i = 0; i < h; i++)
            memcpy(pDataUV + i * w, (unsigned char*)(data->UV + (i * p)), w);

        inferRequest.set_input_tensor(0, inputTensorY);
        inferRequest.set_input_tensor(1, inputTensorUV);
    }

    //-- Infers specified input(s) in synchronous mode
    inferRequest.infer();

    auto outputTensor = inferRequest.get_tensor(outputName);

    // Display class id, bounding box rect info, and confidence % of detected object
    size_t lastDim = outputTensor.get_shape().back();

    if (lastDim == 7) {
        float* data = (float*)outputTensor.data();
        for (size_t i = 0; i < outputTensor.get_size() / lastDim; i++) {
            int imageId = static_cast<int>(data[i * lastDim + 0]);
            int classId = static_cast<int>(data[i * lastDim + 1]);
            float confidence = data[i * lastDim + 2];
            auto x_min = static_cast<int>(data[i * lastDim + 3] * oriWidth);
            auto y_min = static_cast<int>(data[i * lastDim + 4] * oriHeight);
            auto x_max = static_cast<int>(data[i * lastDim + 5] * oriWidth);
            auto y_max = static_cast<int>(data[i * lastDim + 6] * oriHeight);
            if (imageId < 0)
                break;
            if (confidence < 0.5) {
                continue;
            }

            printf("    Class Id (%d),  BBox (%4d, %4d, %4d, %4d),  Confidence (%5.3f)\n",
                classId,
                x_min,
                y_min,
                x_max,
                y_max,
                confidence);
        }
    }

    std::cout << std::endl;
    return;
}

mfxSession CreateVPLSession(mfxLoader* loader, Params* cli) {
    mfxStatus sts = MFX_ERR_NONE;

    // variables used only in 2.x version
    mfxConfig cfg[4];
    mfxVariant cfgVal;
    mfxSession session = NULL;

    //-- Create session
    *loader = MFXLoad();
    VERIFY2(NULL != *loader, "ERROR: MFXLoad failed -- is implementation in path?\n");

    // Implementation used must be the hardware implementation
    cfg[0] = MFXCreateConfig(*loader);
    VERIFY2(NULL != cfg[0], "MFXCreateConfig failed")
        cfgVal.Type = MFX_VARIANT_TYPE_U32;
    cfgVal.Data.U32 = MFX_IMPL_TYPE_HARDWARE;

    sts = MFXSetConfigFilterProperty(cfg[0], (mfxU8*)"mfxImplDescription.Impl", cfgVal);
    VERIFY2(MFX_ERR_NONE == sts, "ERROR: MFXSetConfigFilterProperty failed for Impl");

    // Implementation must provide an HEVC decoder
    cfg[1] = MFXCreateConfig(*loader);
    VERIFY2(NULL != cfg[1], "MFXCreateConfig failed")
        cfgVal.Type = MFX_VARIANT_TYPE_U32;
    cfgVal.Data.U32 = MFX_CODEC_HEVC;
    sts = MFXSetConfigFilterProperty(
        cfg[1],
        (mfxU8*)"mfxImplDescription.mfxDecoderDescription.decoder.CodecID",
        cfgVal);
    VERIFY2(MFX_ERR_NONE == sts, "ERROR: MFXSetConfigFilterProperty failed for decoder CodecID");

    // Implementation used must have VPP scaling capability
    cfg[2] = MFXCreateConfig(*loader);
    VERIFY2(NULL != cfg[2], "MFXCreateConfig failed")
        cfgVal.Type = MFX_VARIANT_TYPE_U32;
    cfgVal.Data.U32 = MFX_EXTBUFF_VPP_SCALING;
    sts = MFXSetConfigFilterProperty(
        cfg[2],
        (mfxU8*)"mfxImplDescription.mfxVPPDescription.filter.FilterFourCC",
        cfgVal);
    VERIFY2(MFX_ERR_NONE == sts, "ERROR: MFXSetConfigFilterProperty failed for VPP scale");

    if (cli->bLegacyGen == false) {
        // Implementation used must provide API version 2.2 or newer
        cfg[3] = MFXCreateConfig(*loader);
        VERIFY2(NULL != cfg[3], "MFXCreateConfig failed")
            cfgVal.Type = MFX_VARIANT_TYPE_U32;
        cfgVal.Data.U32 = VPLVERSION(MAJOR_API_VERSION_REQUIRED, MINOR_API_VERSION_REQUIRED);
        sts = MFXSetConfigFilterProperty(cfg[3],
            (mfxU8*)"mfxImplDescription.ApiVersion.Version",
            cfgVal);
        VERIFY2(MFX_ERR_NONE == sts, "ERROR: MFXSetConfigFilterProperty failed for API version");
    }

    sts = MFXCreateSession(*loader, 0, &session);
    VERIFY2(MFX_ERR_NONE == sts,
        "ERROR: cannot create session -- no implementations meet selection criteria");

    // Print info about implementation loaded
    ShowImplementationInfo(*loader, 0);
    return session;
}