#pragma once
#include <openvino/openvino.hpp>
#include "../core/util.h"
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

int InitVideoSegment(VideoSegParams cliParams);
int SceneDetect(void* pmfxVPPOutSurface);
bool LoadAndReportModel(const std::string& xmlPath,
	ov::Core& core,
	std::shared_ptr<ov::Model>& outModel);
bool CheckOpenVINOEnvironment(ov::Core& core);

class VideoSegmentor {

public:
	VideoSegmentor();
	~VideoSegmentor();

	bool Init(VideoSegParams Params);
	/*
	*  pGenericTexture: points to the frame texture. Windows it is ID3D11Texture2D, Linux it is VASurface 
	*  FrameNo: ID of the incoming generice Texture
	*  bReset: The purpose of the reset is to tell the VideoSegmentor to recalcaute from scratch.
	*/
	int  SceneDetect(void *pGenericTexture, int frameNo, bool bReset );

private:
	void PrintInputAndOutputsInfo(const ov::Model& network);

	ov::InferRequest inferRequest;
	ov::CompiledModel compiledModel;
	ov::Core core;

	std::string m_strInputTensorName;
	std::string m_strOutputTensorName;
};