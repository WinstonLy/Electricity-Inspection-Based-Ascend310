/*
* @Author: winston
* @Date:   2021-03-10 15:41:25
* @Last Modified by:   WinstonLy
* @Last Modified time: 2021-04-07 13:27:59
* @Description: 
* @FilePath: /home/winston/AscendProjects/rtsp_dvpp_infer_dvpp_rtmp_test/atlas200dk_yolov4/Electricity-Inspection-Based-Ascend310/src/image_main.cpp 
*/
#include <iostream>
#include <fstream>
#include <string>
#include <functional>
#include <stdlib.h>
#include <dirent.h>
#include <time.h>
#include <unistd.h>
#include "utils.h"
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include <atomic>
#include <map>
#include <Python.h>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp> 
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
#include "libavutil/time.h"
}

#include "ffmpeg_io.h"
#include "dvpp_vdec.h"
#include "dvpp_venc.h"
#include "dvpp_vpc.h"
#include "sample_objection.h"
#include "model_infer.h"
#include "dvpp_jpege.h"
#include "dvpp_jpegd.h"
// std::function<void(AVPacket *, aclvdecChannelDesc *, uint8_t)> packet_handler;
static uint8_t timeStamp;
static bool runFlag = true;
// static aclrtContext *currentCtx;
// void *picOutBufferDev{nullptr};
fstream resultVdec;
fstream resultVenc;
fstream resultInfer;
fstream resultResize;
fstream resultSend;
fstream resultFFmpeg;
namespace {
  int modelWidth = 608;
  int modelHeight = 608;
  const char* modelPath = "./model/yolov4.om";
  const char* appConf = "./script/object_detection.conf";
}


int main(int argc, char *argv[]){
    // 打开记录处理时间的文件
    resultVdec.open("./data/resultVdec.txt");
    resultVenc.open("./data/resultVenc.txt");
    resultResize.open("./data/resultResize.txt");
    resultInfer.open("./data/resultInfer.txt");
    resultFFmpeg.open("./data/resultFFmpeg.txt");
    resultSend.open("./data/resultSend.txt");

    // //初始化python
    // Py_Initialize();
 
    // //直接运行python代码
    // PyRun_SimpleString("print 'Python Start'");
 
    // //引入当前路径,否则下面模块不能正常导入
    // PyRun_SimpleString("import sys");  
    // PyRun_SimpleString("sys.path.append('./')");  
    // //引入模块
    // PyObject *pModule = PyImport_ImportModule("bin_to_predict_yolov4_pytorch");
    // //获取模块字典属性
    // PyObject *pDict = PyModule_GetDict(pModule);

    // //直接获取模块中的函数
    // PyObject *pFunc = PyObject_GetAttrString(pModule, "post_process");
    
    // 1. AsecndCL init
    const char* aclConfigPath = "./src/acl.json";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl init failed, err code = %d", ret);
        return FAILED;
    }
    ATLAS_LOG_INFO("AscendCL init success");

    // 2. 运行管理资源申请，包括Device、Context、Stream
    uint32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl open device %d failed", deviceId);
        return FAILED;
    }
    ATLAS_LOG_INFO("open device %d success", deviceId);
    
    aclrtContext ctx;
    ret = aclrtCreateContext(&ctx, deviceId);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl create context failed");
        return FAILED;
    }
    CHECK_ACL(aclrtSetCurrentContext(ctx));
    ATLAS_LOG_INFO("create context success");

    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl create stream failed");
        return FAILED;
    }
    ATLAS_LOG_INFO("create stream success");

    // 3. get run mode
    aclrtRunMode runMode;
    ret = aclrtGetRunMode(&runMode);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl get run mode failed");
        return FAILED;
    }
    bool isDevice = (runMode == ACL_DEVICE);
    RunStatus::SetDeviceStatus(isDevice);
    ATLAS_LOG_INFO("acl Device Context Stream init success");
   
   

   
    
    // jpeg decode init
    DvppJpegD processJpegD(stream);
    // dvpp resize 初始化
    DvppVpcResize processVpcResize(stream);
 
 	// 模型初始化
    ModelProcess processModel(stream, modelWidth, modelHeight);

    // const char* modelPath = "./model/yolov3.om";
    processModel.LoadModelFromFileWithMem(argv[1]);
    processModel.CreateDesc();

    Result resCode = processModel.CreateOutputWithMem();
    if(resCode != SUCCESS){
        ATLAS_LOG_ERROR("model infer create output failed");
        processModel.DestroyOutput();
        // CHECK_ACL(acldvppFree(resizeOutputBuffer));
    }


    DvppJpegE processJpegE(stream);
    processJpegE.OpenPresenterChannelImage();
     // loop begin
    PicDesc testPic[] = {
        { "./data/persian_cat_1024_1536_283.jpg", 1024, 1536 },
        { "./data/dog_768_576.jpg", 768, 576 },
        { "./data/eagle_773_512.jpg", 773, 512 },
        { "./data/giraffe_500_500.jpg", 500, 500 },
        { "./data/horses_773_512.jpg", 773, 512 },
        { "./data/person_640_424.jpg", 640, 424 },
        // { "./data/wood_rabbit_1024_1061_330.jpg", 1024, 1061 },
  

   
    };
    for (size_t index = 0; index < sizeof(testPic) / sizeof(testPic[0]); ++index) {
        ATLAS_LOG_INFO("start to process picture:%s", testPic[index].picName.c_str());
    	// 
    	uint32_t devPicBufferSize;
        char *picDevBuffer = GetPicDevBuffer4JpegD(testPic[index], devPicBufferSize);
        if (picDevBuffer == nullptr) {
            ATLAS_LOG_ERROR("get pic device buffer failed,index is %zu", index);
            return FAILED;
        }
        Result ret = processJpegD.InitResource(testPic[index].width, testPic[index].height,
        		                               modelWidth, modelHeight);
        if(ret != SUCCESS){
 			ATLAS_LOG_ERROR("dvpp process failed");
            (void)acldvppFree(picDevBuffer);
            picDevBuffer = nullptr;
            return 1;
 		}
 											   
 		ret = processJpegD.ProcessJpegD(picDevBuffer, devPicBufferSize);
 		if(ret != SUCCESS){
 			ATLAS_LOG_ERROR("dvpp process failed");
            (void)acldvppFree(picDevBuffer);
            picDevBuffer = nullptr;
            return 1;
 		}
 		

        void *dvppOutputBuffer = nullptr;
        int dvppOutputSize;

        processJpegD.GetOutput(&dvppOutputBuffer, dvppOutputSize);

        ATLAS_LOG_INFO("jpegd output size : %d", dvppOutputSize);
        if(dvppOutputBuffer == nullptr){
            ATLAS_LOG_ERROR("get jpegd output failed");
        }
   	 	processVpcResize.Init(testPic[index].width, testPic[index].height, modelWidth, modelHeight);

		processVpcResize.Resize(dvppOutputBuffer, dvppOutputSize); 
        

    	
    	// 创建模型推理的输入输出
    	// int resizeOutputSize = processVpcResize.GetOutputBufferSize();
    	uint32_t* resizeOutputBuffer = (uint32_t*)processVpcResize.GetOutputBuffer();
    	// uint32_t resizeOutputSize = processVpcResize.GetOutputBufferSize();
    	if(resizeOutputBuffer == nullptr){
             ATLAS_LOG_ERROR("get resize output failed");
        }
        uint32_t imageInfoSize;
    	void*    imageInfoBuf;
    	//The second input to Yolov3 is the input image width and height parameter
    	const float imageInfo[4] = {(float)modelWidth, (float)modelHeight, (float)modelWidth, (float)modelHeight};
    	imageInfoSize = sizeof(imageInfo);
    	if (runMode == ACL_HOST)
        	imageInfoBuf = CopyDataHostToDevice((void *)imageInfo, imageInfoSize);  // 实现Host到Device的内存复制
    	else
        	imageInfoBuf = CopyDataDeviceToDevice((void *)imageInfo, imageInfoSize);   // 实现Device内的内存复制
    	if (imageInfoBuf == nullptr) {
        	ATLAS_LOG_ERROR("Copy image info to device failed");
        	return 1;
    	}

    	Result resCode = processModel.CreateInput(resizeOutputBuffer, modelWidth*modelHeight*3/2);
    	if(resCode != SUCCESS){
        	ATLAS_LOG_ERROR("model infer create input failed");
        	processModel.DestroyInput();
        	CHECK_ACL(acldvppFree(resizeOutputBuffer));
        	return FAILED;
    	}


    
        // processJpegE.InitJpegEResource((uint8_t*)processVpcResize.GetInputBuffer(),
        // 								ALIGN_UP128(testPic[index].width), ALIGN_UP128(testPic[index].height));
        processJpegE.InitJpegEResource((uint8_t*)processVpcResize.GetOutputBuffer(),
                                     modelWidth, modelHeight);                               
        processModel.Execute();
        vector<DetectionResult> DetectResults = processModel.PostProcessYolov4(testPic[index].width, testPic[index].height); 

        processJpegE.Process(testPic[index].width, testPic[index].height, DetectResults);
        
        processJpegE.DestroyEncodeResource();
        processJpegD.DestroyJpegDResource();
        processVpcResize.Destroy();
        
        (void)acldvppFree(picDevBuffer);
        picDevBuffer = nullptr;
    }


    // 资源释放 
    processModel.DestroyInput();
    processModel.DestroyOutput();
    processVpcResize.Destroy();
    processJpegD.DestroyJpegDResource();

    ret = aclrtDestroyStream(stream);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("Destroy stream failed, err code = %d", ret);
    }

    ret = aclrtDestroyContext(ctx);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("Destroy context failed, err code = %d", ret);
    }

    ret = aclrtResetDevice(deviceId);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("Destroy context failed, err code = %d", ret);
    }

    CHECK_ACL(aclFinalize());

    // void *res = nullptr;
    // int joinThreadErr = pthread_join(threadId, &res);

    runFlag = false;
    resultVdec.close();
    resultVenc.close();
    resultResize.close();
    resultInfer.close();
    resultFFmpeg.close();
    resultSend.close();

    ATLAS_LOG_INFO("acl final sestroy success");

    return 0;
}

