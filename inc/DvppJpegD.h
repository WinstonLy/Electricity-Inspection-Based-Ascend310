#ifndef __DVPP_JPEGD_H__
#define __DVPP_JPEGD_H__

#pragma once
#include <cstdint>
#include "utils.h"
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include <functional>
extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
}


class DvppJpegD{
public:
	DvppJpegD(aclrtStream _stream);

	~DvppJpegD();

	Result InitResource(uint32_t srcWidth, uint32_t srcHeight, 
						uint32_t modelWidth, uint32_t modelHeight);
	Result ProcessJpegD(void* inputData, uint32_t inputSize);
	void GetOutput(void **outputBuffer, int &outputSize);
	void DestroyJpegDResource();

	void JpegDRegisterHandler(std::function<void(uint8_t *, size_t)> handler);
private:
	aclrtStream stream;
	acldvppChannelDesc* dvppJpegDChannelDesc;

	void* jpegDOutputBuffer;
	acldvppPicDesc* jpegDOutputDesc;
	std::function<void(uint8_t *, size_t)> jpegDHandler;

	uint32_t jpegDOutBufferSize;
};


#endif // __DVPP_JPEGD_H__