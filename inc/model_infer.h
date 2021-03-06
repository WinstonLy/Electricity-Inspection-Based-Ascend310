#ifndef __MODEL_INFER_H__
#define __MODEL_INFER_H__

#include <iostream>
#include "acl/acl.h"
#include "utils.h"
#include <vector>
#include "yolov4_post.h"
#include "ascenddk/presenter/agent/presenter_channel.h"

using namespace ascend::presenter;

#include <functional>

class ModelProcess{
public:
	ModelProcess(aclrtStream _stream, int _width, int _height);
	~ModelProcess();
	
	Result LoadModelFromFileWithMem(const char *modelPath);
	void Unload();

	Result CreateDesc();
	void DestroyDesc();

	Result CreateInput(void* inputDataBuffer, size_t inputBufferSize);

	Result CreateInput(void* inputDataBuffer, size_t inputBufferSize, 
                                 void* imageInfoBuffer, size_t imageInfoSize);
	Result CreateInput(vector<DataInfo>& inputData);

	aclError AddDatasetBuffer(aclmdlDataset *dataset, 
                                        void* buffer, size_t bufferSize);
	void DestroyInput();

	Result CreateOutput();
	Result CreateOutputWithMem();
	void DestroyOutput();

	Result Init(const char* modelPath, void* inputDataBuffer, size_t inputBufferSize);
	void Destroy();

	Result Execute();

	const std::vector<void*> &GetOutputDataBuffers();
    const std::vector<size_t> &GetOutputBufferSizes();

    aclmdlDataset *GetModelOutputData();

    void RegisterHandler(std::function<void(uint8_t*, int)> handler);


    vector<DetectionResult> PostProcessYolov4(int frameWidth, int frameHeight);
    aclError WriteResult(const std::vector<ObjDetectInfo> &objInfos, int frameIndex) const;
    aclError GetObjectInfoYolo(std::vector<RawData> &modelOutput, std::vector<ObjDetectInfo> &objInfos,
    						   int frameWidth, int frameHeight);


    vector<DetectionResult> PostProcessYolov3(int frameWidth, int frameHeight);
    void* GetInferenceOutputItem(uint32_t& itemDataSize,
                                  aclmdlDataset* inferenceOutput, uint32_t idx);
private:
	uint32_t modelId;
	size_t modelMemSize;
	size_t modelWeightSize;

	void* modelMemBuffer;
	void* modelWeightBuffer;
	bool loadFlag; // model load flag


	aclmdlDesc* modelDesc;
	aclmdlDataset* modelInput;
	aclmdlDataset* modelOutput;
	aclrtStream stream;

	int modelWidth;
	int modelHeight;

	std::vector<void *> outputDataBuffers;
    std::vector<size_t> outputDataBufferSizes;
    void* outputBuffer;
	size_t outputBufferSize;
    std::function<void(uint8_t*, int)> bufferHandler;

};


#endif // __MODEL_PROCESS_H__