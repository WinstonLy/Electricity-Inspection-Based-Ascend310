/*
* @Author: winston
* @Date:   2021-01-07 20:58:09
* @Last Modified by:   WinstonLy
* @Last Modified time: 2021-03-31 22:37:14
* @Description: 
* @FilePath: /home/winston/AscendProjects/rtsp_dvpp_infer_dvpp_rtmp_test/atlas200dk_yolov4/Electricity-Inspection-Based-Ascend310/src/ModelProcess.cpp 
*/
#include "ModelProcess.h"
#include <iostream>

// #include <vector>
extern fstream resultInfer;
namespace {
const static std::vector<std::string> yolov3Label = { "person", "bicycle", "car", "motorbike",
"aeroplane","bus", "train", "truck", "boat",
"traffic light", "fire hydrant", "stop sign", "parking meter",
"bench", "bird", "cat", "dog", "horse",
"sheep", "cow", "elephant", "bear", "zebra",
"giraffe", "backpack", "umbrella", "handbag","tie",
"suitcase", "frisbee", "skis", "snowboard", "sports ball",
"kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
"tennis racket", "bottle", "wine glass", "cup",
"fork", "knife", "spoon", "bowl", "banana",
"apple", "sandwich", "orange", "broccoli", "carrot",
"hot dog", "pizza", "donut", "cake", "chair",
"sofa", "potted plant", "bed", "dining table", "toilet",
"TV monitor", "laptop", "mouse", "remote", "keyboard",
"cell phone", "microwave", "oven", "toaster", "sink",
"refrigerator", "book", "clock", "vase","scissors",
"teddy bear", "hair drier", "toothbrush" };
//Inferential output dataset subscript 0 unit is detection box information data
const uint32_t kBBoxDataBufId = 0;
//The unit with subscript 1 is the number of boxes
const uint32_t kBoxNumDataBufId = 1;
//Each field subscript in the box message
enum BBoxIndex { TOPLEFTX = 0, TOPLEFTY, BOTTOMRIGHTX, BOTTOMRIGHTY, SCORE, LABEL };
}

namespace {
const static std::vector<std::string> yolov4Label = {
"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
"truck", "boat", "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", 
"bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", 
"zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
"skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove", 
"skateboard", "surfboard", "tennis_racket", "bottle", "wine_glass", "cup", "fork", 
"knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", 
"carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch", "potted_plant", 
"bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
"cell_phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
"clock", "vase", "scissors", "teddy_bear", "hair_drier", "toothbrush  "};
}

ModelProcess::ModelProcess(aclrtStream _stream, int _width, int _height):stream(_stream), 
    modelWidth(_width), modelHeight(_height), modelId(0), modelMemSize(0),
    modelWeightSize(0),loadFlag(false), modelMemBuffer(nullptr), modelWeightBuffer(nullptr), 
    modelDesc(nullptr), modelInput(nullptr), modelOutput(nullptr), outputBuffer(nullptr), outputBufferSize(0)
{

}
ModelProcess::~ModelProcess(){
    Unload();
    DestroyDesc();
    DestroyInput();
    DestroyOutput();
}

Result ModelProcess::LoadModelFromFileWithMem(const char *modelPath){
	if(loadFlag){
		ATLAS_LOG_ERROR("ACL has already loaded a model");
		return FAILED;
	}
	// 根据模型文件获取模型执行的时候需要的权值内存大小和工作内存大小
    aclError ret = aclmdlQuerySize(modelPath,&modelMemSize, &modelWeightSize);
    if(ret != ACL_ERROR_NONE){
    	ATLAS_LOG_ERROR("ACL query mem and weight size failed, model file is %s", modelPath);
    	return FAILED;
    }

    // 根据权值内存和工作内存大小，申请Device上的工作内存和权值内存
    CHECK_ACL(aclrtMalloc(&modelMemBuffer, modelMemSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&modelWeightBuffer, modelWeightSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // 加载离线模型文件，用户自行管理模型运行内存
    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId, modelMemBuffer,
                                    modelMemSize, modelWeightBuffer, modelWeightSize);
    if(ret != ACL_ERROR_NONE){
    	ATLAS_LOG_ERROR("Load model from file failed, err code = %d", ret);
    	ATLAS_LOG_ERROR("model file path is %s", modelPath);
    	return FAILED;
    }

    loadFlag = true;
    ATLAS_LOG_INFO("load model %s success", modelPath);

    return SUCCESS;
    

}
void ModelProcess::Unload(){
	if(!loadFlag){
		WARN_LOG("no model had been loaded, uload failed");
		return;
	}

	aclError ret = aclmdlUnload(modelId);
	if(ret != ACL_ERROR_NONE){
		ATLAS_LOG_ERROR("ACL unload model failed, modelId is %u", modelId);
	}

    if(modelDesc != nullptr){
    	(void)aclmdlDestroyDesc(modelDesc);
        modelDesc = nullptr;
    }

    if(modelMemBuffer != nullptr){
    	aclrtFree(modelMemBuffer);
    	modelMemBuffer = nullptr;
    	modelMemSize = 0;
    }

    if(modelWeightBuffer != nullptr){
    	aclrtFree(modelWeightBuffer);
    	modelWeightBuffer = nullptr;
    	modelWeightSize  = 0;
    }

    loadFlag = false;

    ATLAS_LOG_INFO("ACL unload model success, modelId is %u", modelId);
}

Result ModelProcess::CreateDesc(){

	// 根据加载成功的模型的ID,获取模型的描述信息
	modelDesc = aclmdlCreateDesc();
	if(modelDesc == nullptr){
		ATLAS_LOG_ERROR("create model description failed");
		return FAILED;
	}

	aclError ret = aclmdlGetDesc(modelDesc, modelId);
	if(ret != ACL_ERROR_NONE){
		ATLAS_LOG_ERROR("get model id failed");
		return FAILED;
	}

	ATLAS_LOG_INFO("create model description success");

	return SUCCESS;

}

void ModelProcess::DestroyDesc(){
	if(modelDesc != nullptr){
		(void)aclmdlDestroyDesc(modelDesc);
		modelDesc = nullptr;
	}
}
Result ModelProcess::CreateInput(void* inputDataBuffer, size_t inputBufferSize){
    vector<DataInfo> inputData = {{inputDataBuffer, inputBufferSize}};
    return CreateInput(inputData);
}

Result ModelProcess::CreateInput(void* inputDataBuffer, size_t inputBufferSize, 
                                 void* imageInfoBuffer, size_t imageInfoSize){
    vector<DataInfo> inputData = {{inputDataBuffer, inputBufferSize},
                                  {imageInfoBuffer, imageInfoSize}};
    return CreateInput(inputData);
}
Result ModelProcess::CreateInput(vector<DataInfo>& inputData){
    if(inputData.size() == 0){
        ATLAS_LOG_ERROR("Create model input failed for no input data");
        return FAILED;
    }

	// 创建aclmdlDataset类型的数据，用于描述模型推理的输入数据，输入的内存地址、内存大小等信息
	modelInput = aclmdlCreateDataset();
	if(modelInput == nullptr){
		ATLAS_LOG_ERROR("can't create dataset, create input failed");
		return FAILED;
	}

    for(uint32_t i = 0; i < inputData.size(); i++){
        aclError ret = AddDatasetBuffer(modelInput, inputData[i].data, inputData[i].size);
        if(ret != ACL_ERROR_NONE){
            ATLAS_LOG_ERROR("Create input failed for add dataset buffer error %d", ret);
            return FAILED;
        }
    }

	ATLAS_LOG_INFO("create model input success");
	return SUCCESS;
}

aclError ModelProcess::AddDatasetBuffer(aclmdlDataset *dataset, 
                                        void* buffer, size_t bufferSize){

    aclDataBuffer* modelDataBuf = aclCreateDataBuffer(buffer, bufferSize);
    if(modelDataBuf == nullptr){
        ATLAS_LOG_ERROR("can't create data buffer, create input failed");
        return ACL_ERROR_MODEL_INPUT_NOT_MATCH;
    }

    aclError ret = aclmdlAddDatasetBuffer(dataset, modelDataBuf);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("can't add dataset buffer, create input failed, err code = %d", ret);
        aclDestroyDataBuffer(modelDataBuf);
        return ret;
    }

    return ACL_ERROR_NONE;
}
void ModelProcess::DestroyInput(){
	if(modelInput == nullptr){
		return;
	}

	for(size_t i = 0; i < aclmdlGetDatasetNumBuffers(modelInput); ++i){
		aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(modelInput, i);
		aclDestroyDataBuffer(dataBuffer);
	}

	aclmdlDestroyDataset(modelInput);
	modelInput = nullptr;
}
// 模型输出需要另行获取
Result ModelProcess::CreateOutput()
{
    if (modelDesc == nullptr) {
        ATLAS_LOG_ERROR("no model description, create ouput failed");
        return FAILED;
    }

    // 创建描述模型推理的输出
    modelOutput = aclmdlCreateDataset();
    if(modelOutput == nullptr){
        ATLAS_LOG_ERROR("can't create dataset, create model output failed");
        return FAILED;
    }

    // 获取模型的输出个数
    size_t outputSize = aclmdlGetNumOutputs(modelDesc);

    // 循环为每个输出申请内存，并将每个输出添加到aclmdlDataset类型的数据中
    for(size_t i = 0; i < outputSize; ++i){
        outputBufferSize = aclmdlGetOutputSizeByIndex(modelDesc, i);
        

        aclError ret = aclrtMalloc(&outputBuffer, outputBufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if(ret != ACL_ERROR_NONE){
            ATLAS_LOG_ERROR("can't malloc buffer, size is %zu, create output failed", outputBufferSize);
            return FAILED;
        }

        aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, outputBufferSize);
        ret = aclmdlAddDatasetBuffer(modelOutput, outputData);
        if(ret != ACL_ERROR_NONE){
            ATLAS_LOG_ERROR("can't add dataset buffer, create output failed");
            aclrtFree(outputBuffer);
            aclDestroyDataBuffer(outputData);
            return FAILED;
        }
    }

    ATLAS_LOG_INFO("create model output success");

    return SUCCESS;
    
}
// 将模型输出存放到数组中
Result ModelProcess::CreateOutputWithMem(){
	if (modelDesc == nullptr) {
        ATLAS_LOG_ERROR("no model description, create ouput failed");
        return FAILED;
    }

	// 创建描述模型推理的输出
	modelOutput = aclmdlCreateDataset();
	if(modelOutput == nullptr){
		ATLAS_LOG_ERROR("can't create dataset, create model output failed");
		return FAILED;
	}

    // 获取模型的输出个数
	size_t outputSize = aclmdlGetNumOutputs(modelDesc);

    // 循环为每个输出申请内存，并将每个输出添加到aclmdlDataset类型的数据中
    for(size_t i = 0; i < outputSize; ++i){
    	size_t bufferSize = aclmdlGetOutputSizeByIndex(modelDesc, i);
    	void* outputBuffer{nullptr};

    	aclError ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    	if(ret != ACL_ERROR_NONE){
    		ATLAS_LOG_ERROR("can't malloc buffer, size is %zu, create output failed", bufferSize);
    		return FAILED;
    	}

     
        outputDataBufferSizes.push_back(bufferSize);
        outputDataBuffers.push_back(outputBuffer);
        std::cout << "create output size : " << bufferSize << std::endl;
        aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, bufferSize);
        ret = aclmdlAddDatasetBuffer(modelOutput, outputData);
        if(ret != ACL_ERROR_NONE){
        	ATLAS_LOG_ERROR("can't add dataset buffer, create output failed");
        	aclrtFree(outputBuffer);
        	aclDestroyDataBuffer(outputData);
        	return FAILED;
        }
    }

    ATLAS_LOG_INFO("create model output success");

    return SUCCESS;
}

void ModelProcess::DestroyOutput(){
    if(modelOutput == nullptr){
    	return;
    }

    for(size_t i = 0; i < aclmdlGetDatasetNumBuffers(modelOutput); ++i){
    	aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(modelOutput, i);
    	void* data = aclGetDataBufferAddr(dataBuffer);
    	(void)aclrtFree(data);
    	(void)aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(modelOutput);
    modelOutput = nullptr;

    if(outputDataBuffers.size() != 0){
        for(auto& j : outputDataBuffers){
            aclrtFree(j);
        }
    }

    if(outputBuffer != nullptr){
        aclrtFree(outputBuffer);
    }

   
}

Result ModelProcess::Execute(){
    clock_t beginTime = clock();
    // 异步接口
	aclError ret = aclmdlExecuteAsync(modelId, modelInput, modelOutput, stream);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("execute model failed, modelId is %u", modelId);
        return FAILED;
    }
    // 阻塞应用程序运行，直到指定stream中的所有任务完成
    ret = aclrtSynchronizeStream(stream);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("synchronize stream failed, err code = %d", ret);
        return FAILED;
    }
    // // 同步接口
    // aclError ret = aclmdlExecute(modelId, modelInput, modelOutput);
    // if (ret != ACL_ERROR_NONE) {
    //     ATLAS_LOG_ERROR("execute model failed, modelId is %u", modelId);
    //     return FAILED;
    // }
    
    clock_t endTime = clock();
    resultInfer << "infer a frame time: " << (double)(endTime - beginTime)*1000/CLOCKS_PER_SEC << " ms" <<endl;

    ATLAS_LOG_INFO("model execute success");
    return SUCCESS;
}

const std::vector<void*> &ModelProcess::GetOutputDataBuffers(){
	return outputDataBuffers;
}

const std::vector<size_t> &ModelProcess::GetOutputBufferSizes() {
  return outputDataBufferSizes;
}


aclmdlDataset *ModelProcess::GetModelOutputData()
{
    return modelOutput;
}


void ModelProcess::RegisterHandler(std::function<void(uint8_t*, int)> handler){
    bufferHandler = handler;
}


void* ModelProcess::GetInferenceOutputItem(uint32_t& itemDataSize,
                                            aclmdlDataset* inferenceOutput ,
                                            uint32_t idx) {

    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(inferenceOutput, idx);
    if (dataBuffer == nullptr) {
        ATLAS_LOG_ERROR("Get the %dth dataset buffer from model "
        "inference output failed", idx);
        
    }

    void* dataBufferDev = aclGetDataBufferAddr(dataBuffer);
    if (dataBufferDev == nullptr) {
        ATLAS_LOG_ERROR("Get the %dth dataset buffer address "
        "from model inference output failed", idx);
        
    }

    size_t bufferSize = aclGetDataBufferSize(dataBuffer);
    if (bufferSize == 0) {
        ATLAS_LOG_ERROR("The %dth dataset buffer size of "
        "model inference output is 0", idx);
        
    }

    void* data = nullptr;
    if (!(RunStatus::GetDeviceStatus())) {
        // data = Utils::CopyDataDeviceToLocal(dataBufferDev, bufferSize);
        // if (data == nullptr) {
        //     ATLAS_LOG_ERROR("Copy inference output to host failed");
        //     return nullptr;
        // }
        ATLAS_LOG_ERROR("is not a device, need copy to device");
    }
    else {
        data = dataBufferDev;
    }

    
    itemDataSize = bufferSize;
    return data;
}

// Yolov3修改之后的模型，加入3个yolo和一个yolodetecouput算子，不需要后处理
vector<DetectionResult> ModelProcess::PostProcess(int frameWidth, 
                                  int frameHeight){
    //Get box information data
    uint32_t dataSize = 0;
    
    // 检测信息按照x上、y上、x下、y下、得分、标签的顺序排列，多个目标也是如此。
    float* detectData = (float*)GetInferenceOutputItem(dataSize, modelOutput, kBBoxDataBufId);
    
    if (detectData == nullptr) 
        ATLAS_LOG_ERROR("detect data get failed");
    //Gets the number of boxes

    uint32_t* boxNum = (uint32_t*)GetInferenceOutputItem(dataSize, modelOutput, kBoxNumDataBufId);
    
    if (boxNum == nullptr)
        ATLAS_LOG_ERROR("get infer output box num failed");

    //Number of boxes The first data is valid
    
    uint32_t totalBox = boxNum[0];
    //
    float widthScale  = (float)(frameWidth) / modelWidth;
    float heightScale = (float)(frameHeight) / modelHeight;
    ATLAS_LOG_INFO("width Scale %f, heightScale %f", widthScale, heightScale);
    ATLAS_LOG_INFO("totalBox : %d", totalBox);
    vector<DetectionResult> detectResults;
    for (uint32_t i = 0; i < totalBox; i++) {
        DetectionResult oneResult;
        Point point_lt, point_rb;
        //get the confidence of the detected object. Anything less than 0.8 is considered invalid
        uint32_t score = (uint32_t)(detectData[totalBox * SCORE + i] * 100);
        if (score < 80) continue;
        ATLAS_LOG_INFO("infer score %d", score);
        //get the frame coordinates and converts them to the coordinates on the original frame
        oneResult.lt.x = detectData[totalBox * TOPLEFTX + i] * widthScale;
        oneResult.lt.y = detectData[totalBox * TOPLEFTY + i] * heightScale;
        oneResult.rb.x = detectData[totalBox * BOTTOMRIGHTX + i] * widthScale;
        oneResult.rb.y = detectData[totalBox * BOTTOMRIGHTY + i] * heightScale;
        //Construct a string that marks the object: object name + confidence
        uint32_t objIndex = (uint32_t)detectData[totalBox * LABEL + i];
        oneResult.result_text = yolov3Label[objIndex] + std::to_string(score) + "\%";
        ATLAS_LOG_INFO("%d %d %d %d %s\n", oneResult.lt.x, oneResult.lt.y,
         oneResult.rb.x, oneResult.rb.y, oneResult.result_text.c_str());
        

        detectResults.emplace_back(oneResult);
    }

    //If it is the host side, the data is copied from the device and the memory used by the copy is freed
    if (!(RunStatus::GetDeviceStatus())) {
        delete[]((uint8_t*)detectData);
        delete[]((uint8_t*)boxNum);
    }

    // //Sends inference results and images to presenter Server for display
    // SendImage(detectResults, frameData, frameWidth, frameHeight);

    return detectResults;
}


vector<DetectionResult> ModelProcess::PostProcess(){
    std::vector<RawData> Output;
    // std::vector<void*> inferOutputBuffers = GetOutputDataBuffers();
    // std::vector<size_t> inferOutputSizes = GetOutputBufferSizes();
    int size = outputDataBuffers.size();
    for(int i = 0; i < size; ++i){
        RawData rawData = RawData();
        rawData.data.reset(outputDataBuffers[i], [](void*) {});
        rawData.lenOfByte = outputDataBufferSizes[i];
        Output.push_back(std::move(rawData));
    }

    size_t outputLength = Output.size();
    if(outputLength <= 0){
        ATLAS_LOG_ERROR("Failed to get model infer output data");
        // return ACL_ERROR_MEMORY_ADDRESS_UNALIGNED;
    }

    std::vector<ObjDetectInfo> objInfos;
    static int frameIndex = 0;
    aclError ret = GetObjectInfoTensorflow(Output, objInfos);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("Falied to get TensorFlow model output, ret = %d", ret);
        // return ret;
    }
    vector<DetectionResult> detectResults;
    for(int i = 0; i < objInfos.size(); i++){
        DetectionResult oneResult;
        Point point_lt, point_rb;
        //get the confidence of the detected object. Anything less than 0.8 is considered invalid
        uint32_t score = (uint32_t)(objInfos[i].confidence * 100);
        if (score < 80) continue;
        ATLAS_LOG_INFO("infer score %d", score);
        //get the frame coordinates and converts them to the coordinates on the original frame
        oneResult.lt.x = objInfos[i].leftTopX;
        oneResult.lt.y = objInfos[i].leftTopY;
        oneResult.rb.x = objInfos[i].rightBotX;
        oneResult.rb.y = objInfos[i].rightBotY;
        //Construct a string that marks the object: object name + confidence
        uint32_t objIndex = (uint32_t)objInfos[i].classId;
        oneResult.result_text = yolov4Label[objIndex] + std::to_string(score) + "\%";
        ATLAS_LOG_INFO("%d %d %d %d %s\n", oneResult.lt.x, oneResult.lt.y,
         oneResult.rb.x, oneResult.rb.y, oneResult.result_text.c_str());
        

        .emplace_back(oneResult);
    }
 
    

    // // 手动提取3个feature，用python文件解析
    // static int imageNum = 0;
    // int tensorNum = 1;
    // for(int i = 0; i < outputDataBuffers.size(); ++i){
    //     std::string fileNameSave = "./results/output" + std::to_string(imageNum) + "_" 
    //                                 + std::to_string(tensorNum ) + ".bin";
    //     FILE* output = fopen(fileNameSave.c_str(), "wb+");
        
    //     void* data = outputDataBuffers[i];
    //     int sizeData = outputDataBufferSizes[i];
    //     size_t num = fwrite(data, 1, sizeData, output);
    //     std::cout << "write sizeData 0 : " << num << std::endl;
        
        
        
    //     fclose(output);

    //     if(tensorNum == 3){
    //         break;
    //     }
    //     ++tensorNum;

    // }
    // ++imageNum;
   

   



    

    return detectResults;
    // ret = WriteResult(objInfos, frameIndex);
    // if(ret != ACL_ERROR_NONE){
    //     ATLAS_LOG_ERROR("Failed to wrtie result, ret = %d", ret);
    //     return ret;
    // }
    // ++frameIndex;

    // return ACL_ERROR_NONE;

}

aclError ModelProcess::WriteResult(const std::vector<ObjDetectInfo> &objInfos, int frameIndex)
    const
{
    std::string timeString;
    GetCurTimeString(timeString);
    // Create result file under result directory
    // SetFileDefaultUmask();
    
    std::string resultName = "./out/result/result_" + std::to_string(frameIndex);
    std::string fileNameSave = "./out/result/result_" + std::to_string(frameIndex);

    std::ofstream tfile(resultName, std::ios::app);
    // Check result file validity
    if (tfile.fail()) {
        ATLAS_LOG_ERROR("Failed to open result file: %s", resultName.c_str());
        return ACL_ERROR_MEMORY_ADDRESS_UNALIGNED;
    }
    tfile.seekp(0, tfile.end);
    size_t dstFileSize = tfile.tellp();
    tfile.close();
    // if (dstFileSize > FILE_SIZE) {
    //     if (access(resultBakName_.c_str(), 0) == APP_ERR_OK) {
    //         APP_ERROR ret = remove(resultBakName_.c_str());
    //         if (ret != APP_ERR_OK) {
    //             LogError << "remove " << resultBakName_ << " failed." << std::endl;
    //             return ret;
    //         }
    //     }
    //     APP_ERROR ret = rename(resultName.c_str(), resultBakName_.c_str());
    //     if (ret != APP_ERR_OK) {
    //         LogError << "rename " << resultName << " failed." << std::endl;
    //         return ret;
    //     }
    // }
    tfile.open(resultName, std::ios::out);
    if (tfile.fail()) {
       ATLAS_LOG_ERROR("Failed to open result file: %s", resultName.c_str());
        return ACL_ERROR_MEMORY_ADDRESS_UNALIGNED;
    }
    tfile << "[Date:" << timeString << " Frame:" << frameIndex
          << "] Object detected number is " << objInfos.size() << std::endl;
    // Write inference result into file
    std::cout <<"[Date:" << timeString <<" Frame:" << frameIndex
          << "] Object detected number is " << objInfos.size() << std::endl;
    
    for (uint32_t i = 0; i < objInfos.size(); i++) {
        tfile << "#Obj" << i << ", " << "box(" << objInfos[i].leftTopX << ", " << objInfos[i].leftTopY << ", "
              << objInfos[i].rightBotX << ", " << objInfos[i].rightBotY << ") "
              << " confidence: " << objInfos[i].confidence << "  lable: " << objInfos[i].classId << std::endl;
    }

    // for (uint32_t i = 0; i < objInfos.size(); i++) {
    //     std::cout << "#Obj" << i << ", " << "box(" << objInfos[i].leftTopX << ", " << objInfos[i].leftTopY << ", "
    //           << objInfos[i].rightBotX << ", " << objInfos[i].rightBotY << ") "
    //           << " confidence: " << objInfos[i].confidence << "  lable: " << objInfos[i].classId << std::endl;
    // }
                           
    tfile << std::endl;
    tfile.close();
    return ACL_ERROR_NONE;
}


aclError ModelProcess::GetObjectInfoTensorflow(std::vector<RawData> &modelOutput, std::vector<ObjDetectInfo> &objInfos)
{
    std::vector<std::shared_ptr<void>> hostPtr;
    
    for (size_t i = 0; i < modelOutput.size(); i++) {
        void *hostPtrBuffer = outputDataBuffers[i];
        std::shared_ptr<void> hostPtrBufferManager(hostPtrBuffer, [](void *) {});
        aclError ret = aclrtMemcpy(hostPtrBuffer, modelOutput[i].lenOfByte, modelOutput[i].data.get(),
            modelOutput[i].lenOfByte, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret!= ACL_ERROR_NONE || hostPtrBuffer == nullptr) {
            ATLAS_LOG_ERROR("Failed to copy output buffer of model from device to host, ret =%d", ret);
            return ret;
        }
        hostPtr.push_back(hostPtrBufferManager);
    }

    YoloImageInfo yoloImageInfo = {};
    Yolov3DetectionOutput(hostPtr, objInfos, yoloImageInfo);
    return ACL_ERROR_NONE;
}





