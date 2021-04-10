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

* File utils.h
* Description: handle file operations
*/
#pragma once

#include <iostream>
#include <vector>
#include <unistd.h>
#include <dirent.h>
#include <fstream>
#include <memory>
#include <cstring>
#include <map>
#include <mutex>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <stdio.h>
#include <string>
#include <errno.h>
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

using namespace std::chrono;

using namespace std;

#define AVC1_TAG 0x31637661

#define ATLAS_LOG_INFO(fmt, args...)  fprintf(stdout, "[INFO]   [Func]%s  [Line]%d: " fmt "\n", __func__, __LINE__, ##args)
#define WARN_LOG(fmt, args...)  fprintf(stdout, "[WARN]   [Func]%s  [Line]%d: " fmt "\n", __func__, __LINE__, ##args)
#define ATLAS_LOG_ERROR(fmt, args...) fprintf(stdout, "[ERROR]  [Func]%s  [Line]%d: " fmt "\n", __func__, __LINE__, ##args)

#define RGBU8_IMAGE_SIZE(width, height) ((width) * (height) * 3)
#define YUV420SP_SIZE(width, height) ((width) * (height) * 3 / 2)

#define ALIGN_UP(num, align) (((num) + (align) - 1) & ~((align) - 1))
#define ALIGN_UP2(num) ALIGN_UP(num, 2)
#define ALIGN_UP16(num) ALIGN_UP(num, 16)
#define ALIGN_UP128(num) ALIGN_UP(num, 128)

#define PERF_TIMER()                                                           \
  auto __CONCAT__(temp_perf_obj_, __LINE__) =                                  \
      PerfTimer(__FILE__, __LINE__, __FUNCTION__)
      
#define SHARED_PRT_DVPP_BUF(buf) (shared_ptr<uint8_t>((uint8_t *)(buf), [](uint8_t* p) { acldvppFree(p); }))
#define SHARED_PRT_U8_BUF(buf) (shared_ptr<uint8_t>((uint8_t *)(buf), [](uint8_t* p) { delete[](p); }))

#define CHECK_ACL(x)                                                           \
  do {                                                                         \
    aclError __ret = x;                                                        \
    if (__ret != ACL_ERROR_NONE) {                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret        \
                << std::endl;                                                  \
    }                                                                          \
  } while (0);

template<class Type>
std::shared_ptr<Type> MakeSharedNoThrow() {
    try {
        return std::make_shared<Type>();
    }
    catch (...) {
        return nullptr;
    }
}

/* @brief generate shared pointer of memory
 * @param [in] buf memory pointer, malloc by new
 * @return shared pointer of input buffer
 */
#define SHARED_PRT_U8_BUF(buf) (shared_ptr<uint8_t>((uint8_t *)(buf), [](uint8_t* p) { delete[](p); }))

#define MAKE_SHARED_NO_THROW(memory, memory_type) \
    do { \
            memory = MakeSharedNoThrow<memory_type>(); \
    }while(0);

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
}Result;

typedef struct PicDesc {
    std::string picName;
    int width;
    int height;
} PicDesc;

struct DataInfo {
    void* data;
    size_t size;
};

struct Resolution {
    uint32_t width = 0;
    uint32_t height = 0;
};

struct InferenceOutput {
    void* data = nullptr;
    uint32_t size;
};
// Description of data in device
struct RawData {
    size_t lenOfByte; // Size of memory, bytes
    // std::shared_ptr<void> data; // Smart pointer of data
    void* data;
};

// struct Point {
//     std::uint32_t x;
//     std::uint32_t y;
// };

// struct DetectionResult {
//     Point lt;   //The coordinate of left top point
//     Point rb;   //The coordinate of the right bottom point
//     std::string result_text;  // Face:xx%
// };



struct ImageData {
    acldvppPixelFormat format;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t alignWidth = 0;
    uint32_t alignHeight = 0;
    uint32_t size = 0;
    std::shared_ptr<uint8_t> data = nullptr;
};

struct Rect {
    uint32_t ltX = 0;
    uint32_t ltY = 0;
    uint32_t rbX = 0;
    uint32_t rbY = 0;
};

struct BBox {
    Rect rect;
    uint32_t score;
    string text;
};

class RunStatus {
public:
    static void SetDeviceStatus(bool _isDevice)
    {
        isDevice = _isDevice;
    }
    static bool GetDeviceStatus()
    {
        return isDevice;
    }
private:
    RunStatus() {}
    ~RunStatus() {}
    static bool isDevice;
};

int align_up(int size, int align);
int yuv420sp_size(int h, int w);
bool WriteToFile(const char *fileName, void *dataDev, int dataSize, bool flag);
acldvppStreamFormat h264_ffmpeg_profile_to_acl_stream_fromat(int profile);
void* CopyDataDeviceToDevice(void* deviceData, uint32_t dataSize);
void* CopyDataHostToDevice(void* deviceData, uint32_t dataSize);
void* CopyDataToDevice(void* data, uint32_t dataSize, aclrtMemcpyKind policy);
char* GetPicDevBuffer4JpegD(const PicDesc &picDesc, uint32_t &devPicBufferSize);
void* ReadBinFile(std::string fileName, uint32_t &fileSize);
void GetCurTimeString(std::string &timeString);