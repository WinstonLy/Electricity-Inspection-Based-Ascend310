#ifndef __DVPP_VENC_H__
#define __DVPP_VENC_H__

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include <pthread.h>
#include <thread>

#include "utils.h"
#include "ascenddk/presenter/agent/presenter_channel.h"
#include <functional>

using namespace ascend::presenter;

class DvppVenc{
public:
	DvppVenc(aclrtStream& _stream);
	~DvppVenc();
	void Destroy();
	Result Init(const pthread_t pthread_id, int height, int width);
    aclError SendVencFrame(vector<DetectionResult>& detectionResults,uint8_t* frameData);
    aclError SendVencFrame(uint8_t* frameData, size_t size);

    void RegisterHandler(std::function<void(uint8_t *, int)> handler);
    const std::function<void(uint8_t *, int)> &GetHandler();
    int GetVencWidth();
    int GetVencHeight();

    Result OpenPresenterChannel();
    Result SendImageDisplay(vector<DetectionResult>& detectionResults, uint8_t* data, int size);
private:
	int vencHeight;
	int vencWidth;
	int vencSize;

    bool isDevice;
	aclvencChannelDesc *vencChannelDesc;
    aclvencFrameConfig *vencFrameConfig;
    acldvppPicDesc *vencInputPicDesc;
    aclrtStream stream;

    Channel* channel;

    std::function<void(uint8_t*, int)> vencBufferHandler;
};

#endif // __DVPP_VENC_H__