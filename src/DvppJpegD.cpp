/*
* @Author: winston
* @Date:   2021-03-10 14:39:43
* @Last Modified by:   winston
* @Last Modified time: 2021-03-18 10:17:53
* @Description: 
* @FilePath: /home/winston/AscendProjects/rtsp_dvpp_infer_dvpp_rtmp_test/atlas200dk_yolov4/atlas200dk_yolov4_test/src/DvppJpegD.cpp 
*/
#include "DvppJpegD.h"
#include <iostream>
#include "utils.h"


DvppJpegD::DvppJpegD(aclrtStream _stream)
	: stream(_stream), dvppJpegDChannelDesc(nullptr),jpegDOutBufferSize(0),
	  jpegDOutputDesc(nullptr), jpegDOutputBuffer(nullptr)
{

}

DvppJpegD::~DvppJpegD(){
	DestroyJpegDResource();
}

Result DvppJpegD::InitResource(uint32_t srcWidth, uint32_t srcHeight, 
							   uint32_t modelWidth, uint32_t modelHeight){
	dvppJpegDChannelDesc = acldvppCreateChannelDesc();
	if(dvppJpegDChannelDesc == nullptr){
		ATLAS_LOG_ERROR("acldvppCreateChannelDesc failed");
		return FAILED;
	}

	aclError ret = acldvppCreateChannel(dvppJpegDChannelDesc);
	if(ret != ACL_ERROR_NONE){
		ATLAS_LOG_ERROR("acldvppCreateChannel failed, err code = %d", ret);
		return FAILED;
	}

	uint32_t widthAlignment = 128;
    uint32_t heightAlignment = 16;
    uint32_t sizeAlignment = 3;
    uint32_t sizeNum = 2;
    uint32_t decodeOutWidthStride = ALIGN_UP128(srcWidth);
    uint32_t decodeOutHeightStride = ALIGN_UP16(srcHeight);
    if (decodeOutWidthStride == 0 || decodeOutHeightStride == 0) {
        ATLAS_LOG_ERROR("InitDecodeOutputDesc AlignmentHelper failed");
        return FAILED;
    }

    jpegDOutBufferSize = decodeOutWidthStride * decodeOutHeightStride * sizeAlignment / sizeNum;
    ret = acldvppMalloc(&jpegDOutputBuffer, jpegDOutBufferSize);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acldvppMalloc jpegDOutputBuffer failed, ret = %d", ret);
        return FAILED;
    }

    jpegDOutputDesc = acldvppCreatePicDesc();
    if (jpegDOutputDesc == nullptr) {
        ATLAS_LOG_ERROR("acldvppCreatePicDesc jpegDOutputDesc failed");
        return FAILED;
    }

    CHECK_ACL(acldvppSetPicDescData(jpegDOutputDesc, jpegDOutputBuffer));
    CHECK_ACL(acldvppSetPicDescFormat(jpegDOutputDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
    CHECK_ACL(acldvppSetPicDescWidth(jpegDOutputDesc, srcWidth));
    CHECK_ACL(acldvppSetPicDescHeight(jpegDOutputDesc, srcHeight));
    CHECK_ACL(acldvppSetPicDescWidthStride(jpegDOutputDesc, decodeOutWidthStride));
    CHECK_ACL(acldvppSetPicDescHeightStride(jpegDOutputDesc, decodeOutHeightStride));
    CHECK_ACL(acldvppSetPicDescSize(jpegDOutputDesc, jpegDOutBufferSize));

    ATLAS_LOG_INFO("process jpegd init success");
    return SUCCESS;
}

Result DvppJpegD::ProcessJpegD(void* inputData, uint32_t inputSize){
	aclError ret = acldvppJpegDecodeAsync(dvppJpegDChannelDesc, reinterpret_cast<void *>(inputData),
        inputSize, jpegDOutputDesc, stream);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acldvppJpegDecodeAsync failed, aclRet = %d", ret);
        return FAILED;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("decode aclrtSynchronizeStream failed, aclRet = %d", ret);
        return FAILED;
    }

    ATLAS_LOG_INFO("process jpegd success");

    return SUCCESS;
}

void DvppJpegD::JpegDRegisterHandler(std::function<void(uint8_t *, size_t)> handler){
	jpegDHandler = handler;
}

void DvppJpegD::DestroyJpegDResource(){
	if(jpegDOutputDesc != nullptr){
		acldvppDestroyPicDesc(jpegDOutputDesc);
        jpegDOutputDesc = nullptr;
	}
	if (dvppJpegDChannelDesc != nullptr) {
        aclError ret = acldvppDestroyChannel(dvppJpegDChannelDesc);
        if (ret != ACL_ERROR_NONE) {
            ATLAS_LOG_ERROR("acldvppDestroyChannel failed, aclRet = %d", ret);
        }

        (void)acldvppDestroyChannelDesc(dvppJpegDChannelDesc);
        dvppJpegDChannelDesc = nullptr;
    }

    if(jpegDOutputBuffer != nullptr){
    	acldvppFree(jpegDOutputBuffer);
        jpegDOutputBuffer = nullptr;
    }
}

void DvppJpegD::GetOutput(void **outputBuffer, int &outputSize)
{
    *outputBuffer = jpegDOutputBuffer;
    outputSize = jpegDOutBufferSize;
    jpegDOutputBuffer = nullptr;
    jpegDOutputBuffer = 0;
}