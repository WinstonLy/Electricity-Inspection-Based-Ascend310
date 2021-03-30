/*
* @Author: winston
* @Date:   2021-01-09 10:32:05
* @Last Modified by:   WinstonLy
* @Last Modified time: 2021-03-30 10:30:19
* @Description: 
* @FilePath: /home/winston/AscendProjects/rtsp_dvpp_infer_dvpp_rtmp_test/atlas200dk_yolov4/Electricity-Inspection-Based-Ascend310/src/SampleProcess.cpp 
*/
#include "SampleProcess.h"

static bool runFlag = true;

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

SampleProcess::SampleProcess():deviceId(0), threadId(0),
    ctx(nullptr), runMode(ACL_HOST), isDevice(false)
{

}
SampleProcess::~SampleProcess(){
	DestroyResource();
}

void SampleProcess::DestroyResource(){
    aclError ret = aclrtDestroyStream(stream);
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

    void *res = nullptr;
    int joinThreadErr = pthread_join(threadId, &res);

    runFlag = false;
    
    ATLAS_LOG_INFO("acl final sestroy success");
}
Result SampleProcess::AclInit(uint8_t _deviceId){
    // 1. AsecndCL init
    const char* aclConfigPath = "../src/acl.json";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl init failed, err code = %d", ret);
        return FAILED;
    }
    ATLAS_LOG_INFO("AscendCL init success");

    // 2. 运行管理资源申请，包括Device、Context、Stream
    deviceId = _deviceId;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl open device %d failed", deviceId);
        return FAILED;
    }
    ATLAS_LOG_INFO("open device %d success", deviceId);

    ret = aclrtCreateContext(&ctx, deviceId);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl create context failed");
        return FAILED;
    }
    CHECK_ACL(aclrtSetCurrentContext(ctx));
    ATLAS_LOG_INFO("create context success");

    ret = aclrtCreateStream(&stream);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl create stream failed");
        return FAILED;
    }
    ATLAS_LOG_INFO("create stream success");

    // 3. get run mode
    ret = aclrtGetRunMode(&runMode);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acl get run mode failed");
        return FAILED;
    }
    isDevice = (runMode == ACL_DEVICE);

    ATLAS_LOG_INFO("acl Device Context Stream init success");
    return SUCCESS;
}
Result SampleProcess::VdecProcess(){
   
}
Result SampleProcess::FFMPEGInProcess(string rtspInput){
}
Result SampleProcess::FFMPEGOutProcess(){

}
Result SampleProcess::ResizeProcess(){

}
Result SampleProcess::VencProcess(){

}
Result SampleProcess::ModelInferProcess(){

}
Result SampleProcess::CreateThread(){
	// 接口指定处理Stream上回调函数的线程，回调函数和线程是由用户调用aclvdecSetChannelDesc系列接口时指定的
    uint8_t createThreadErr = pthread_create(&threadId, nullptr, ThreadFunc, nullptr);
    if (createThreadErr != 0) {
        ATLAS_LOG_ERROR("create thread failed, err = %d",  createThreadErr);
        return FAILED;
    }

    ATLAS_LOG_INFO("create thread successfully, threadId = %lu", threadId);

    CHECK_ACL(aclrtSubscribeReport(static_cast<uint64_t>(threadId), stream));
}
pthread_t SampleProcess::GetPthreadId(){
	return threadId;
}

    

