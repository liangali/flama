//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#ifndef TOOLS_CLI_VAL_SURFACE_SHARING_SRC_VPP_H_
#define TOOLS_CLI_VAL_SURFACE_SHARING_SRC_VPP_H_

#include <memory>
#include <vector>
#include "../device/hw_device.h"
#include "../utils/util.h"

#ifdef TOOLS_ENABLE_OPENCL
    #include "../device/process_frames_ocl.h"
#endif

class CVPPTest : public CTest {
public:
    CVPPTest();
    virtual ~CVPPTest();
    CVPPTest(const CVPPTest &)            = delete;
    CVPPTest &operator=(const CVPPTest &) = delete;

    mfxStatus Init(int tIndex,
                   Options &opts,
                   std::vector<mfxU32> *adapterNumbers = nullptr) override;
    mfxStatus Run() override;
    mfxStatus ProcessingFrame(void *inputTexture, void *outputTexture);

private:
    int m_tIndex;
    mfxLoader m_loader;
    mfxSession m_session;
    mfxU32 m_frameNum = 0;
    DevCtx *m_pDevCtx;
#ifdef TOOLS_ENABLE_OPENCL
    OpenCLCtx *m_pOclCtx;
#endif
    mfxVideoParam m_vppParams;
    Options        m_opts{}; 
    Options *m_pOpts;
    FrameInfo m_frameInfo;
    FileInfo m_fileInfo;
    mfxMemoryInterface *m_memoryInterface;

    mfxStatus CreateVPLSession();
    mfxStatus ProcessStreamVPP();

};

#endif // TOOLS_CLI_VAL_SURFACE_SHARING_SRC_VPP_H_

