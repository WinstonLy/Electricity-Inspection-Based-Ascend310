/*
* @Author: WinstonLy
* @Date:   2021-04-02 20:36:33
* @Last Modified by:   WinstonLy
* @Last Modified time: 2021-04-14 20:18:39
* @Description: 
* @FilePath: /home/winston/AscendProjects/rtsp_dvpp_infer_dvpp_rtmp_test/atlas200dk_yolov4/Electricity-Inspection-Based-Ascend310/src/multi_main.cpp 
*/

// 利用多线程来完成处理任务，测试阶段
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
#include <queue>
#include <thread>
#include <mutex>
#include <sys/time.h>
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
bool runFlag = false;
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
// 视频流参数
static int width;
static int height;
static acldvppStreamFormat enType;
static acldvppPixelFormat format;

typedef std::pair<int, RawData> imgPair;
class pariComp{
public:
	bool operator()(const imgPair& img1, const imgPair& img2) const{
		if(img1.first == img2.first)	return img1.first > img2.first;
		return img1.first > img2.first;
	}
};

aclrtStream stream;

// 解码、推理、后处理的三个信号量

std::mutex mtxQueueInput;
std::mutex mtxQueueInfer;
std::mutex mtxQueueJpege;
std::mutex mtxQueueDectec;
// 存储三个线程的共享数据队列

std::queue<RawData> queueInput;
std::queue<std::pair<vector<size_t>, vector<void*>> > queueInfer;
std::queue<RawData> queueJpege;


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

// get now time
double whatIsTimeNow(){
	struct timeval time;
	if(gettimeofday(&time, NULL)){
		return 0;
	}

	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

// ffmepg get rtsp 线程
void VideoDecode(const char* rtspPath){
    runFlag = true;
    // 绑定cpu
    cpu_set_t mask;
    int cpuId = 0;
    
    // 初始化为0
    CPU_ZERO(&mask);
    // 设置CPU
    CPU_SET(cpuId, &mask);
    // 绑定CPU
    if(pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0){
        std::cout << "[ERROR] [FUNC]"<< __FUNCTION__  << "[LINE]: "<< __LINE__ 
                  << "set thread affinity failed" << std::endl;
    }
    std::cout << "[INFO} bind ffmpeg run to CPU " << cpuId << std::endl;

    // 1. AsecndCL init
    const char* aclConfigPath = "./src/acl.json";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl init failed, err code = %d", ret);
        return;
    }
    ATLAS_LOG_INFO("AscendCL init success");

    // 2. 运行管理资源申请，包括Device、Context、Stream
    uint32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl open device %d failed", deviceId);
        return ;
    }
    ATLAS_LOG_INFO("open device %d success", deviceId);
    
    aclrtContext ctx;
    ret = aclrtCreateContext(&ctx, deviceId);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl create context failed");
        return;
    }
    CHECK_ACL(aclrtSetCurrentContext(ctx));
    ATLAS_LOG_INFO("create context success");


    ret = aclrtCreateStream(&stream);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl create stream failed");
        runFlag = false;
        return;
    }
    ATLAS_LOG_INFO("create stream success");

    // 3. get run mode
    aclrtRunMode runMode;
    ret = aclrtGetRunMode(&runMode);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl get run mode failed");
        return;
    }
    bool isDevice = (runMode == ACL_DEVICE);
    RunStatus::SetDeviceStatus(isDevice);
    ATLAS_LOG_INFO("acl Device Context Stream init success");

    

    // 接口指定处理Stream上回调函数的线程，回调函数和线程是由用户调用aclvdecSetChannelDesc系列接口时指定的
    pthread_t threadId;
    uint8_t createThreadErr = pthread_create(&threadId, nullptr, ThreadFunc, nullptr);
    if (createThreadErr != 0) {
        ATLAS_LOG_ERROR("create thread failed, err = %d",  createThreadErr);
        return;
    }

    ATLAS_LOG_INFO("create thread successfully, threadId = %lu", threadId);

    CHECK_ACL(aclrtSubscribeReport(threadId, stream));

    // 初始化ffmpeg
    FFMPEGInput processInput;
    processInput.InputInit(rtspPath);

    // 获取视频流信息
    width               = processInput.GetWidth();
    height              = processInput.GetHeight();
    acldvppStreamFormat enType  = processInput.GetProfile();
    acldvppPixelFormat format   = PIXEL_FORMAT_YUV_SEMIPLANAR_420;

    // dvpp dec 初始化
    DvppVdec processVdec; 
    processVdec.Init(threadId, height, width, enType, format);
    // processVdec.SetDeviceCtx(&ctx);

    // 初始化,ffmpeg和dec的连接接口
    processInput.RegisterHandler(
        [&](AVPacket* packet) { processVdec.vdecSendFrame(packet); });

    processInput.Run();

    // 资源释放 
    processInput.Destroy();
    processVdec.Destroy();

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
}

// 预处理+infer的线程
void ResizeAndInfer(){
    // bind cpu 
    cpu_set_t mask;
    int cpuId = 1;
    sleep(3); 
    CPU_ZERO(&mask);
    CPU_SET(cpuId, &mask);

    if(pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0){
        std::cout << "[ERROR] [FUNC]"<< __FUNCTION__  << "[LINE]: "<< __LINE__ 
                  << "set thread affinity failed" << std::endl;
    }
    std::cout << "[INFO} bind resize and infer to CPU " << cpuId << std::endl;

    aclrtContext ctx;
    aclError ret = aclrtCreateContext(&ctx, 0);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl create context failed");
        return;
    }
    CHECK_ACL(aclrtSetCurrentContext(ctx));
    ATLAS_LOG_INFO("create context success");


    ret = aclrtCreateStream(&stream);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl create stream failed");
        runFlag = false;
        return;
    }
    ATLAS_LOG_INFO("create stream success");

    // dvpp resize 初始化
    DvppVpcResize processVpcResize(stream);
    processVpcResize.Init(width, height, modelWidth, modelHeight);

    // 模型初始化
    ModelProcess modelInfer(stream, modelWidth, modelHeight);
    modelInfer.Init(modelPath, processVpcResize.GetOutputBuffer(), modelWidth*modelHeight*3/2);
   
    
    while(true){
        mtxQueueInput.lock();
        if(queueInput.empty()){
            mtxQueueInput.unlock();
            // usleep(100);
            continue;
        }
        else{
            RawData inputData = queueInput.front();
            queueInput.pop();
            mtxQueueInput.unlock();
            
            processVpcResize.Resize(inputData.data, inputData.lenOfByte);
            modelInfer.Execute();
            
            mtxQueueJpege.lock();
            queueJpege.push(inputData);
            mtxQueueJpege.unlock();
        }
    }

    //  释放资源
    processVpcResize.Destroy();
    modelInfer.Destroy();

    ret = aclrtDestroyStream(stream);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("Destroy stream failed, err code = %d", ret);

    }

    ret = aclrtDestroyContext(ctx);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("Destroy context failed, err code = %d", ret);
    }

    ret = aclrtResetDevice(0);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("Destroy context failed, err code = %d", ret);
    }
}

void PostAndJpege(int modelWidth, int modelHeight){
    // bind cpu 
    cpu_set_t mask;
    int cpuId = 3;
    sleep(4); 
    CPU_ZERO(&mask);
    CPU_SET(cpuId, &mask);

    if(pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0){
        std::cout << "[ERROR] [FUNC]"<< __FUNCTION__  << "[LINE]: "<< __LINE__ 
                  << "set thread affinity failed" << std::endl;
    }
    std::cout << "[INFO} bind post and jpege to CPU " << cpuId << std::endl;

    aclrtContext ctx;
    aclError ret = aclrtCreateContext(&ctx, 0);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl create context failed");
        return;
    }
    CHECK_ACL(aclrtSetCurrentContext(ctx));
    ATLAS_LOG_INFO("create context success");


    ret = aclrtCreateStream(&stream);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl create stream failed");
        runFlag = false;
        return;
    }
    ATLAS_LOG_INFO("create stream success");
    
    // 初始化dvpp jpege
    DvppJpegE processJpegE(stream);
    processJpegE.OpenPresenterChannelVideo();
    while(true){
        mtxQueueJpege.lock();
        if(queueJpege.empty()){
            mtxQueueJpege.unlock();
            continue;
        }
        mtxQueueInfer.lock();
        if(queueInfer.empty()){
            mtxQueueInfer.unlock();
            continue;
        }
        
        RawData jpegeData = queueJpege.front();
        queueJpege.pop();
        mtxQueueJpege.unlock();

        processJpegE.InitJpegEResource((uint8_t*)jpegeData.data, width, height);

    
        vector<size_t> sizeData  = queueInfer.front().first;
        vector<void*> bufferData = queueInfer.front().second;
        
        queueInfer.pop();
        mtxQueueInfer.unlock();

        vector<DetectionResult> DetectResults = PostProcessYolov4(sizeData, bufferData, width, height, modelWidth, modelHeight);

    
        processJpegE.Process(width, height, DetectResults);

        aclError ret = acldvppFree(jpegeData.data);
        if(ret != ACL_ERROR_NONE){
            ATLAS_LOG_ERROR("free input jepge failed");
        }
        processJpegE.DestroyEncodeResource();  
    }

    ret = aclrtDestroyStream(stream);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("Destroy stream failed, err code = %d", ret);

    }

    ret = aclrtDestroyContext(ctx);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("Destroy context failed, err code = %d", ret);
    }

    ret = aclrtResetDevice(0);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("Destroy context failed, err code = %d", ret);
    }
   
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

    std::thread t1(VideoDecode, argv[1]);
    std::thread t2(ResizeAndInfer);
    std::thread t3(PostAndJpege, modelWidth, modelHeight);

    
    t1.join();
    t2.join();
    t3.join(); 

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
