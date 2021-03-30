#ifndef __DVPP_VDEC_H__
#define __DVPP_VDEC_H__

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include <pthread.h>
#include <thread>


extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
}

#include <functional>
#include <fstream>
#include <time.h>
#include "utils.h"

class DvppVdec{
public:
    DvppVdec();
    ~DvppVdec();
    void Destroy();

    Result Init(const pthread_t threadId, int height, int width, 
    	        acldvppStreamFormat _enType, acldvppPixelFormat _format);
    Result vdecSendFrame(AVPacket* pkt);
    void RegisterHandler(std::function<void(uint8_t *, size_t)> handler);
     
    void SetDeviceCtx(aclrtContext *ctx);
    aclrtContext *GetDeviceCtx();
    const std::function<void(uint8_t *, size_t)> &GetHandler();

private:
	int vdecHeight;
	int vdecWidth;
	uint32_t outputSize;
	int timeStamp;


	aclvdecChannelDesc *vdecChannelDesc;
	// aclvdecFrameConfig *vdecFrameConfig;
	acldvppStreamFormat enType;
	acldvppPixelFormat format;
	aclrtContext *devCtx;
	std::function<void(uint8_t *, size_t)> vdecBufferHandler;
};


#endif // __DVPP_VDEC_H__
