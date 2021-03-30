#ifndef __MODEL_PROCESS_H__
#define __MODEL_PROCESS_H__

#include <iostream>
#include "acl/acl.h"
#include "utils.h"
#include <vector>
#include "YoloPostProcess.h"
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

	Result Execute();

	const std::vector<void*> &GetOutputDataBuffers();
    const std::vector<size_t> &GetOutputBufferSizes();

    aclmdlDataset *GetModelOutputData();

    void RegisterHandler(std::function<void(uint8_t*, int)> handler);


    vector<DetectionResult> PostProcess();
    aclError WriteResult(const std::vector<ObjDetectInfo> &objInfos, int frameIndex) const;
    aclError GetObjectInfoTensorflow(std::vector<RawData> &modelOutput, std::vector<ObjDetectInfo> &objInfos);


    vector<DetectionResult> PostProcess(int frameWidth, int frameHeight);
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