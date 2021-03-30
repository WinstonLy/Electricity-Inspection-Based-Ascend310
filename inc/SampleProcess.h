#ifndef __SAMPLE_PROCESS_H__
#define __SAMPLE_PROCESS_H__
#include <iostream>
#include "acl/acl.h"
#include <pthread.h>
#include <thread>
#include "utils.h"

class SampleProcess{
public:
    SampleProcess();
    ~SampleProcess();
    Result AclInit(uint8_t _deviceId);
    Result FFMPEGInProcess(string rtspInput);
    Result FFMPEGOutProcess();
    Result ResizeProcess();
    Result VencProcess();
    Result ModelInferProcess();
    Result VdecProcess();
    Result CreateThread();
    pthread_t GetPthreadId();
    void DestroyResource();


private:
    uint8_t deviceId;
    pthread_t threadId;
    bool isDevice;
    aclrtContext ctx;
    aclrtStream stream;
    aclrtRunMode runMode;
};

#endif // __SAMPLE_PROCESS_H__