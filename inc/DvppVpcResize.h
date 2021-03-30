#ifndef __DVPP_VPC_RESIZE_H__
#define __DVPP_VPC_RESIZE_H__

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

#include <functional>
#include <fstream>
#include <time.h>
#include "utils.h"

class DvppVpcResize{
public:
	DvppVpcResize(aclrtStream _stream);
	~DvppVpcResize();
	void Destroy();

	aclError Init(int srcWidth, int srcHeight, int dstWidth, int dstHeight);
	Result Resize(void* pdata, size_t size);

	void RegisterHandler(std::function<void(uint8_t *)> handler);
	uint8_t *GetOutputBuffer();
    int GetOutputBufferSize();

    void GetSrcData(ImageData& frameData);
    uint8_t* GetInputBuffer();

private:
	int outputBufferSize;
    int inputBufferSize;

    void *resizeInputBuffer;
    void *resizeOutputBuffer;

    acldvppPicDesc *resizeInputPicDesc;
    acldvppPicDesc *resizeOutputPicDesc;
    acldvppChannelDesc *resizeChannelDesc;
    acldvppResizeConfig *resizeConfig;
    aclrtStream stream;

    std::function<void(uint8_t *)> bufferHandler;
// public: 
//     ImageData srcData;
};


#endif // __DVPP_VPC_RESIZE__