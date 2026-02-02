//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

#ifndef TOOLS_CLI_VAL_SURFACE_SHARING_SRC_UTIL_H_
#define TOOLS_CLI_VAL_SURFACE_SHARING_SRC_UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ostream>
#include <string>
#include <vector>

#include "./defs.h"
#include <openvino/genai/continuous_batching_pipeline.hpp>

#ifdef USE_MEDIASDK1
#include "mfxvideo.h"
enum {
    MFX_FOURCC_I420 = MFX_FOURCC_IYUV /*!< Alias for the IYUV color format. */
};
#else
#include "vpl/mfxjpeg.h"
#include "vpl/mfxvideo.h"
#endif

#if (MFX_VERSION >= 2000)
#include "vpl/mfxdispatcher.h"
#endif

#if defined(_WIN32) || defined(_WIN64)

    #include <atlbase.h>
    #include <conio.h>
    #include <d3d11.h>
    #include <dxgi.h>
    #include <dxgi1_2.h>
    #include <windows.h>
    #include <windowsx.h>

#elif defined __linux__

    #include <fcntl.h>
    #include <unistd.h>

    #include "va/va.h"
    #include "va/va_drm.h"

    #include "va/va_drmcommon.h"

#endif

// VPL utility constants
#define WAIT_100_MILLISECONDS 100
#define MAX_PATH              260
#define MAX_WIDTH             3840
#define MAX_HEIGHT            2160
#define IS_ARG_EQ(a, b)       (!strcmp((a), (b)))

#define ALIGN16(value)           (((value + 15) >> 4) << 4)
#define ALIGN32(X)               (((mfxU32)((X) + 31)) & (~(mfxU32)31))
#define VPLVERSION(major, minor) (major << 16 | minor)

#define VERIFY(x, y)               \
    if (!(x)) {                    \
        throw std::logic_error(y); \
    }

#define VERIFY2(x, y)      \
    if (!(x)) {            \
        printf("%s\n", y); \
        return NULL;       \
    }

typedef struct _VideoSegParams {
    mfxIMPL impl;
#if (MFX_VERSION >= 2000)
    mfxVariant implValue;
#endif

    // similarity detection algorithm
    unsigned int windowSize;
    float        threshold;

    // inference model configruation
    unsigned int inferWidth;
    unsigned int inferHeight;
    char* inmodelName;

    bool bZeroCopy;

#if defined(_WIN32) || defined(_WIN64)
    ID3D11Device* pD3D11Device;
#else
    VADisplay lvaDisplay;
#endif
} VideoSegParams;
#define TSTRING2STRING(tstr) tstr
// decode-vpp.cpp
int RunDecodeVPP(Options *opts, FileInfo *fileInfo);

// encode.cpp
int RunEncode(Options *opts, FileInfo *fileInfo);

// util.cpp
void Usage(void);
bool ParseArgsAndValidate(int argc, char *argv[], Options *opts);
mfxStatus GetAdaptersInfo(Options *opts, bool bPrint = false);
void ShowTestInfo(Options *opts);
const char *FourCCToString(mfxU32 fourCC);
mfxStatus ReadEncodedStream(mfxBitstream &bs, std::ifstream &f);
void WriteEncodedStream(mfxBitstream &bs, std::ofstream &f);
#ifdef _WIN32
mfxStatus WriteRawFrame(ID3D11Device *pDevice, ID3D11Texture2D *pTex2D, std::ofstream &f);
#else
mfxStatus WriteRawFrame(VADisplay vaDisplay, VASurfaceID vaSurfaceID, std::ofstream &f);
#endif
mfxStatus WriteRawFrame(mfxFrameSurface1 *surface, std::ofstream &f);
mfxStatus ReadRawFrame(FrameInfo fi, mfxU8 *bs, std::ifstream &f);
mfxStatus ReadRawFrame(mfxFrameSurface1 *surface, std::ifstream &f);
bool CheckKB_Quit(void);

std::string WideToUtf8(const std::wstring& ws);
std::wstring Utf8ToWide(const std::string& s);
enum class FSPolicy {
    FrameInterval,     
    TimeWindowQuota,    
    Mixed,              
    KeyframePriority,   
    MixedKeyframe
};

struct FSConfig {
    FSPolicy policy = FSPolicy::FrameInterval;
    int frame_interval = 60;          
    double window_seconds = 1.0;      
    int max_per_window = 2;           
    int min_frames_between = 0;       
    double min_seconds_between = 0.0; 
    int max_cached = 64;              
    bool remove_after_process = true; 
    // Keyframe / SceneCut
    bool force_keyframe = true;      
    bool enable_scene_cut = false;     
    int min_frames_between_scene_cut = 0;
    double min_seconds_between_scene_cut = 0.0;
    bool enable_cache = false;
};

struct BatchConfig {
    int batch_trigger = 10;
    int max_cache = 128;
    int cb_batch_size = 10;
    int decode_window = 0;
    bool flush_partial = true;
    bool new_batch_mode = false; 
    int max_frames_per_request = 128;
};


extern BatchConfig g_batchConfig; 

struct CommonConfig {  
    bool debug = false;
    bool use_cb = false;
    bool cb_multi_thread = false; // enable CB engine/statistics threads when using CB
    bool new_multithread = false; // run ProcessCBQueueAndReport in a sub thread
    bool hw_decode = true; // decode_mode == "hw" or "hardware"
    std::string input_video_path;
    std::string log_path = "D:/logs";
    int vpp_down_width = 224;  // from common.vpp_downscaling.width
    int vpp_down_height = 224; // from common.vpp_downscaling.height
    std::string prompt_video = "请用一句话精确的描述这个视频中的主要内容， 必须30字以内";
};

extern CommonConfig g_commonConfig;
extern FSConfig g_fsConfig;

struct VLMConfig {
    std::string path;
    std::string device;
    bool enable_continuous_batching = false;
    ov::genai::SchedulerConfig shedulerConfig;
};
extern VLMConfig g_vlmConfig;

// ============================================================================
// VPL Utility Functions (from legacy util.hpp)
// ============================================================================

// VPL Utility Types
enum ExampleParams { PARAM_IMPL = 0, PARAM_INFILE, PARAM_INRES, PARAM_COUNT };
enum ParamGroup {
    PARAMS_CREATESESSION = 0,
    PARAMS_DECODE,
    PARAMS_ENCODE,
    PARAMS_VPP,
    PARAMS_TRANSCODE,
    PARAMS_DECVPP
};

typedef struct _Params {
    mfxIMPL impl;
#if (MFX_VERSION >= 2000)
    mfxVariant implValue;
#endif

    char* infileName;
    char* inmodelName;

    mfxU16 srcWidth;
    mfxU16 srcHeight;

    bool bZeroCopy;
    bool bLegacyGen;
} Params;

// VPL Utility Functions
char* ValidateFileName(char* in);
bool ValidateSize(char* in, mfxU16* vsize, mfxU32 vmax);
bool ParseArgsAndValidate(int argc, char* argv[], Params* params, ParamGroup group);
void* InitAcceleratorHandle(mfxSession session, int* fd);
void FreeAcceleratorHandle(void* accelHandle, int fd);
void ShowImplementationInfo(mfxLoader loader, mfxU32 implnum);
mfxU32 GetSurfaceSize(mfxU32 FourCC, mfxU32 width, mfxU32 height);
int GetFreeSurfaceIndex(mfxFrameSurface1* SurfacesPool, mfxU16 nPoolSize);
mfxStatus AllocateExternalSystemMemorySurfacePool(mfxU8** buf,
                                                   mfxFrameSurface1* surfpool,
                                                   mfxFrameInfo frame_info,
                                                   mfxU16 surfnum);
void FreeExternalSystemMemorySurfacePool(mfxU8* dec_buf, mfxFrameSurface1* surfpool);
mfxStatus ReadEncodedStream(mfxBitstream& bs, FILE* f);
void PrintInputAndOutputsInfo(const ov::Model& network);

#endif // TOOLS_CLI_VAL_SURFACE_SHARING_SRC_UTIL_H_
