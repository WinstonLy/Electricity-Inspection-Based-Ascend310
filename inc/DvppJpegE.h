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

* File dvpp_process.h
* Description: handle dvpp process
*/
#ifndef __DVPP_JPEGE_H__
#define __DVPP_JPEGE_H__
#pragma once
#include <cstdint>

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "utils.h"
#include <vector>
#include <fstream>
#include <time.h>
#include "ascenddk/presenter/agent/presenter_channel.h"
#include "ascenddk/presenter/agent/presenter_types.h"
using namespace ascend::presenter;

class DvppJpegE{
public:
    /**
    * @brief Constructor
    * @param [in] stream: stream
    */
    DvppJpegE(aclrtStream &_stream);

    /**
    * @brief Destructor
    */
    ~DvppJpegE();

    /**
    * @brief process encode
    * @return result
    */
    aclError Process(size_t width, size_t height, vector<DetectionResult>& detectionResults);

   /**
    * @brief release encode resource
    */
    void DestroyEncodeResource();
    void DestroyResource();
    aclError InitJpegEResource(uint8_t* inputImage, size_t width, size_t height);
    aclError InitEncodeInputDesc(uint8_t* srcYuvImage, size_t width, size_t height);
    // void DestroyResource();
    // void DestroyOutputPara();
    Result OpenPresenterChannelVideo();
    Result OpenPresenterChannelImage();
    Result SendImageDisplay(vector<DetectionResult>& detectionResults);
    size_t GetJpegeOutputSize();
private:
    aclrtStream stream;
    acldvppChannelDesc* jpegeChannelDesc;

    acldvppJpegeConfig* jpegeConfig;

    uint32_t encodeOutBufferSize;
    void* encodeOutBufferDev; // encode output buffer
    acldvppPicDesc* encodeInputDesc; //encode input desc
    Channel* channel;

    size_t jpegeWidth;
    size_t jpegeHeight;

};

#endif // DVPP_JPEGE_H__