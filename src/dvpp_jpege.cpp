/**
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File dvpp_process.cpp
* Description: handle dvpp process
*/

#include <iostream>
#include "acl/acl.h"
#include "dvpp_jpege.h"

using namespace std;
extern fstream resultVenc;
extern fstream resultSend;
DvppJpegE::DvppJpegE(aclrtStream& _stream): stream(_stream), jpegeChannelDesc(nullptr), 
                    jpegeConfig(nullptr), encodeOutBufferSize(0), encodeOutBufferDev(nullptr), 
                    encodeInputDesc(nullptr), channel(nullptr), jpegeWidth(0), jpegeHeight(0)
{

}


DvppJpegE::~DvppJpegE() {
    delete channel;
    channel = nullptr;
}

aclError DvppJpegE::InitEncodeInputDesc(uint8_t* inputImage, size_t width, size_t height)
{
    

    // 创建jpege channel描述信息
    jpegeChannelDesc = acldvppCreateChannelDesc();
    if (jpegeChannelDesc == nullptr) {
        ATLAS_LOG_ERROR("Create jpege channel desc failed");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    aclError ret = acldvppCreateChannel(jpegeChannelDesc);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("acldvppCreateChannel failed, aclRet = %d", ret);
        return ret;
    }

    uint32_t alignWidth = ALIGN_UP16(width);
    uint32_t alignHeight = ALIGN_UP2(height);
    if (alignWidth == 0 || alignHeight == 0) {
        ATLAS_LOG_ERROR("Input image width %zu or height %zu invalid", width, height);
        return ACL_ERROR_FORMAT_NOT_MATCH;
    }
    uint32_t inputBufferSize = YUV420SP_SIZE(alignWidth, alignHeight);

    encodeInputDesc = acldvppCreatePicDesc();
    if (encodeInputDesc == nullptr) {
        ATLAS_LOG_ERROR("Create dvpp pic desc failed");
        return ACL_ERROR_INTERNAL_ERROR;
    }
   
    CHECK_ACL(acldvppSetPicDescData(encodeInputDesc, 
                          reinterpret_cast<void *>(inputImage)));
    CHECK_ACL(acldvppSetPicDescFormat(encodeInputDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
    CHECK_ACL(acldvppSetPicDescWidth(encodeInputDesc, width));
    CHECK_ACL(acldvppSetPicDescHeight(encodeInputDesc, height));
    CHECK_ACL(acldvppSetPicDescWidthStride(encodeInputDesc, alignWidth));
    CHECK_ACL(acldvppSetPicDescHeightStride(encodeInputDesc, alignHeight));
    CHECK_ACL(acldvppSetPicDescSize(encodeInputDesc, inputBufferSize));


    // //Connect the presenter server
    // ret = OpenPresenterChannel();
    // if (ret != ACL_ERROR_NONE) {
    //     ATLAS_LOG_ERROR("Open presenter channel failed");
    //     return ret;
    // }

    ATLAS_LOG_INFO("Dvpp JpegE InitEncodeInputDesc success");

    return ACL_ERROR_NONE;
}

aclError DvppJpegE::InitJpegEResource(uint8_t* srcYuvImage, size_t width, size_t height) {
    if(srcYuvImage == nullptr){
        ATLAS_LOG_ERROR("==========secYuvImage is null=========");
    }
    uint32_t encodeLevel = 100; // default optimal level (0-100)

    aclError ret = InitEncodeInputDesc(srcYuvImage, width, height);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("Dvpp jpege init input desc failed, err code = %d", ret);
        return ret;  
    }
	
    jpegeConfig = acldvppCreateJpegeConfig();
    CHECK_ACL(acldvppSetJpegeConfigLevel(jpegeConfig, encodeLevel));

    CHECK_ACL(acldvppJpegPredictEncSize(encodeInputDesc, jpegeConfig, &encodeOutBufferSize));
    ret = acldvppMalloc(&encodeOutBufferDev, encodeOutBufferSize);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("Malloc dvpp memory error(%d)", ret);
        return ret;
    }

    ATLAS_LOG_INFO("Dvpp JpegE InitJpegEResource success");

    return ACL_ERROR_NONE;
}

aclError DvppJpegE::Process(size_t width, size_t height, vector<DetectionResult>& detectionResults)
{
    jpegeWidth = width;
    jpegeHeight = height;
   
    clock_t beginTime = clock();
    aclError ret = acldvppJpegEncodeAsync(jpegeChannelDesc, 
                                 encodeInputDesc, 
                                 encodeOutBufferDev, 
                                 &encodeOutBufferSize, 
                                 jpegeConfig, stream);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("Dvpp jpege async failed, error:%d", ret);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("Dvpp jpege sync stream failed, error:%d", ret);
        return ret;
    }
    clock_t endTime = clock();
    resultVenc << "venc a frame time: " << (double)(endTime - beginTime)*1000/CLOCKS_PER_SEC << " ms" <<endl;
        
    // static int countImage = 0;
    // std::string file = "./results/output" + std::to_string(countImage) + ".jpg";
    // bool flag = false;
    // WriteToFile(file.c_str(), encodeOutBufferDev, encodeOutBufferSize, flag);
    // ++countImage;
 
    SendImageDisplay(detectionResults);
    return ACL_ERROR_NONE;
}

void DvppJpegE::DestroyEncodeResource()
{
    if (jpegeConfig != nullptr) {
        (void)acldvppDestroyJpegeConfig(jpegeConfig);
        jpegeConfig = nullptr;
    }

    if (encodeInputDesc != nullptr) {
        (void)acldvppDestroyPicDesc(encodeInputDesc);
        encodeInputDesc = nullptr;
    }

    if (jpegeChannelDesc != nullptr) {
        aclError aclRet = acldvppDestroyChannel(jpegeChannelDesc);
        if (aclRet != ACL_ERROR_NONE) {
            ATLAS_LOG_ERROR("Destroy dvpp channel error: %d", aclRet);
        }

        (void)acldvppDestroyChannelDesc(jpegeChannelDesc);
        jpegeChannelDesc = nullptr;
    }
    if(encodeOutBufferDev != nullptr){
        acldvppFree(encodeOutBufferDev);
        encodeOutBufferDev = nullptr;
    }

    // delete channel;
    // channel = nullptr;

    ATLAS_LOG_INFO("dvpp jpege destroy success");
}
void DvppJpegE::DestroyResource(){
    DestroyEncodeResource();

    delete channel;
    channel = nullptr;
}
size_t DvppJpegE::GetJpegeOutputSize(){
    return encodeOutBufferSize;
}

Result DvppJpegE::OpenPresenterChannelVideo() {
    OpenChannelParam param;
    param.host_ip = "192.168.1.223";  //IP address of Presenter Server
    param.port = 7006;  //port of present service
    param.channel_name = "video";
    param.content_type = ContentType::kVideo;  //content type is Video
    ATLAS_LOG_INFO("OpenChannel start");
    PresenterErrorCode errorCode = OpenChannel(channel, param);
    ATLAS_LOG_INFO("OpenChannel param");
    if (errorCode != PresenterErrorCode::kNone) {
        ATLAS_LOG_ERROR("OpenChannel failed %d", static_cast<int>(errorCode));
        return FAILED;
    }
    ATLAS_LOG_INFO("OpenChannel success");                              
    return SUCCESS;
}
Result DvppJpegE::OpenPresenterChannelImage() {
    OpenChannelParam param;
    param.host_ip = "192.168.1.223";  //IP address of Presenter Server
    param.port = 7006;  //port of present service
    param.channel_name = "image";
    param.content_type = ContentType::kImage;  //content type is Video
    ATLAS_LOG_INFO("OpenChannel start");
    PresenterErrorCode errorCode = OpenChannel(channel, param);
    ATLAS_LOG_INFO("OpenChannel param");
    if (errorCode != PresenterErrorCode::kNone) {
        ATLAS_LOG_ERROR("OpenChannel failed %d", static_cast<int>(errorCode));
        return FAILED;
    }
    ATLAS_LOG_INFO("OpenChannel success");                              
    return SUCCESS;
}

Result DvppJpegE::SendImageDisplay(vector<DetectionResult>& detectionResults) 
{
    clock_t beginTime = clock();
    ATLAS_LOG_INFO("start send image dispaly");
    ImageFrame imageParam;
    imageParam.format = ImageFormat::kJpeg;
    imageParam.width = jpegeWidth;;
    imageParam.height = jpegeHeight;
    imageParam.size = encodeOutBufferSize;
    imageParam.data = (uint8_t*)encodeOutBufferDev;
    imageParam.detection_results = detectionResults;

    // ATLAS_LOG_INFO("imageParam width = %d, height = %d, size = %d", imageParam.width, imageParam.height, imageParam.size);
    //Sends the detected object frame information and frame image to the Presenter Server for display
    PresenterErrorCode errorCode = PresentImage(channel, imageParam);
    if (errorCode != PresenterErrorCode::kNone) {
        ATLAS_LOG_ERROR("PresentImage failed %d", static_cast<int>(errorCode));
        return FAILED;
    }
    clock_t endTime = clock();
    resultSend << "send a frame time: " << (double)(endTime - beginTime)*1000/CLOCKS_PER_SEC << " ms" <<endl;
    ATLAS_LOG_INFO("send img to presenter server success");
    return SUCCESS;
}





