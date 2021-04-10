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


// 处理回调函数的线程
static void *ThreadFunc(void *arg)
{
    // Notice: create context for this thread
    int deviceId = 0;
    aclrtContext context = nullptr;
    aclError ret = aclrtCreateContext(&context, deviceId);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("aclrtCreateContext failed, err code %d",  ret);
        return ((void*)(-1));
    }

    ATLAS_LOG_INFO("thread start ");
    while (runFlag) {
        // Notice: timeout 1000ms
        aclError aclRet = aclrtProcessReport(1000);
    }

    ret = aclrtDestroyContext(context);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("aclrtDestroyContext failed, ret=%d",  ret);
    }

    return (void*)0;
}
int main(int argc, char *argv[]){
	// check the input when the application execcte
	if((argc < 2) || (argv[1] == nullptr)){
		ATLAS_LOG_ERROR("Please input: ./main <rtsp_dir>");
		return FAILED;
	}

  
    // 打开记录处理时间的文件
    resultVdec.open("./results/resultVdec.txt", ios::out);
    resultVenc.open("./results/resultVenc.txt", ios::out);
    resultResize.open("./results/resultResize.txt", ios::out);
    resultInfer.open("./results/resultInfer.txt", ios::out);
    resultFFmpeg.open("./results/resultFFmpeg.txt", ios::out);
    resultSend.open("./results/resultSend.txt", ios::out);
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
   
    // 接口指定处理Stream上回调函数的线程，回调函数和线程是由用户调用aclvdecSetChannelDesc系列接口时指定的
    pthread_t threadId;
    uint8_t createThreadErr = pthread_create(&threadId, nullptr, ThreadFunc, nullptr);
    if (createThreadErr != 0) {
        ATLAS_LOG_ERROR("create thread failed, err = %d",  createThreadErr);
        return FAILED;
    }

    ATLAS_LOG_INFO("create thread successfully, threadId = %lu", threadId);

    CHECK_ACL(aclrtSubscribeReport(threadId, stream));

    // 初始化ffmpeg
    FFMPEGInput processInput;
    processInput.InputInit(string(argv[1]));
    // dvpp dec 初始化
    DvppVdec processVdec; 

    int width               = processInput.GetWidth();
    int height              = processInput.GetHeight();
    acldvppStreamFormat enType  = processInput.GetProfile();
    acldvppPixelFormat format   = PIXEL_FORMAT_YUV_SEMIPLANAR_420;

    processVdec.Init(threadId, height, width, enType, format);
    processVdec.SetDeviceCtx(&ctx);

    // 初始化,ffmpeg和dec的连接接口
    processInput.RegisterHandler(
        [&](AVPacket* packet) { processVdec.vdecSendFrame(packet); });

    // dvpp resize 初始化
    DvppVpcResize processVpcResize(stream);
    processVpcResize.Init(width, height, modelWidth, modelHeight);

    // 初始化，vdec和resize的接口
    processVdec.RegisterHandler(
        [&](uint8_t* buffer, size_t size) { processVpcResize.Resize(buffer, size); });

    // 模型初始化
    ModelProcess processModel(stream, modelWidth, modelHeight);

    // const char* modelPath = "./model/yolov3.om";
    processModel.LoadModelFromFileWithMem(modelPath);
    processModel.CreateDesc();
    // 创建模型推理的输入输出
    uint32_t* resizeOutputBuffer = (uint32_t*)processVpcResize.GetOutputBuffer();
    // uint32_t resizeOutputSize = processVpcResize.GetOutputBufferSize();
    
    // yolov3的第二个输入
    // uint32_t imageInfoSize;
    // void*    imageInfoBuf;
    // //The second input to Yolov3 is the input image width and height parameter
    // const float imageInfo[4] = {(float)modelWidth, (float)modelHeight, (float)modelWidth, (float)modelHeight};
    // imageInfoSize = sizeof(imageInfo);
    // if (runMode == ACL_HOST)
    //     imageInfoBuf = CopyDataHostToDevice((void *)imageInfo, imageInfoSize);  // 实现Host到Device的内存复制
    // else
    //     imageInfoBuf = CopyDataDeviceToDevice((void *)imageInfo, imageInfoSize);   // 实现Device内的内存复制
    // if (imageInfoBuf == nullptr) {
    //     ATLAS_LOG_ERROR("Copy image info to device failed");
    //     return FAILED;
    // }

    Result resCode = processModel.CreateInput(resizeOutputBuffer, modelWidth*modelHeight*3/2 
                                              );
    if(resCode != SUCCESS){
        ATLAS_LOG_ERROR("model infer create input failed");
        processModel.DestroyInput();
        CHECK_ACL(acldvppFree(resizeOutputBuffer));
        return FAILED;
    }

    resCode = processModel.CreateOutputWithMem();
    if(resCode != SUCCESS){
        ATLAS_LOG_ERROR("model infer create output failed");
        processModel.DestroyOutput();
        CHECK_ACL(acldvppFree(resizeOutputBuffer));
    }

    // 初始化 dvpp venc
    // DvppVenc processVenc(stream);
    // 编码resize之后的数据
    // processVenc.Init(threadId, modelHeight, modelWidth);
    
    // 编码resize之前的数据
    // processVenc.Init(threadId, height, width);
    
    // 初始化，resize和model infer的接口
    // 测试视频编码
    // processVpcResize.RegisterHandler([&] (uint8_t* buffer){ 

    //     processModel.Execute();
    //     vector<DetectionResult> DetectResults = processModel.PostProcess();
    //     // 视频编码resize之后的数据
    //     // processVenc.SendVencFrame(DetectResults, processVpcResize.GetOutputBuffer());
        
    //     // 视频编码resize之前的数据
    //     processVenc.SendVencFrame(DetectResults, processVpcResize.GetInputBuffer());
    //     });

    // // 初始化输出ffmpeg
    // FFMPEGOutput processOutput;
    // processOutput.Init(height, width, processInput.GetFrame(), AV_PIX_FMT_NV12, string(argv[2]));

    // // 初始化venc和ffmpeg output的接口
    // processVenc.RegisterHandler([&] (uint8_t* buffer, int size){
    //     processOutput.SendRtmpFrame(buffer, size); });
    

    // 初始化dvpp jpege
    DvppJpegE processJpegE(stream);
    processJpegE.OpenPresenterChannelVideo();
    processVpcResize.RegisterHandler([&] (uint8_t* buffer){     
        processJpegE.InitJpegEResource((uint8_t*)processVpcResize.GetInputBuffer(), width, height);
        processModel.Execute();

        // yolov3 post
        // vector<DetectionResult> DetectResults = processModel.PostProcessYolov3(width, height); 
        
        // yolov4 post
       
        vector<DetectionResult> DetectResults = processModel.PostProcessYolov4(width, height);
       
        processJpegE.Process(width, height, DetectResults);

        processJpegE.DestroyEncodeResource();
         });
       
    

    processInput.Run();

    // 资源释放 
    processVdec.Destroy();
    processInput.Destroy();
    processVpcResize.Destroy();
    
    // processVenc.Destroy();
    // processOutput.Destroy(); 


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
