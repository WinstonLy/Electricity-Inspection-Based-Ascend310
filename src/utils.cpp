/*
* @Author: winston
* @Date:   2021-01-11 16:28:43
* @Last Modified by:   winston
* @Last Modified time: 2021-03-15 21:32:02
* @Description: 
* @FilePath: /home/winston/AscendProjects/rtsp_dvpp_infer_dvpp_rtmp_test/atlas200dk_yolov4/atlas200dk_yolov4_test/src/utils.cpp 
*/
#include "utils.h"
#include <ctime>
#include <time.h>
bool RunStatus::isDevice = false;

int align_up(int size, int align) {
  return (size + (align - 1)) / align * align;
}

int yuv420sp_size(int h, int w) { return (h * w * 3) / 2; }

/**
 * [WriteToFile description]
 * @param  fileName 保存文件名
 * @param  dataDev  待保存数据
 * @param  dataSize 数据大小
 * @param  flag     true:YUV, false:H264
 * @return          [description]
 */
bool WriteToFile(const char *fileName, void *dataDev, int dataSize, bool flag)
{
    if (dataDev == nullptr) {
        ATLAS_LOG_ERROR("dataDev is nullptr!");
        return false;
    }

    // copy output to host memory
    void *data = nullptr;
    aclError aclRet;
    if (!(RunStatus::GetDeviceStatus())) {
         std::cout << "is a device, don't cpoy to host" << std::endl;
        data = malloc(dataSize);
        if (data == nullptr) {
          ATLAS_LOG_ERROR("malloc host data buffer failed. dataSize=%u\n",  dataSize);
          return false;
        }
        aclRet = aclrtMemcpy(data, dataSize, dataDev, dataSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (aclRet != ACL_ERROR_NONE) {
            ATLAS_LOG_ERROR("acl memcpy data to host failed, dataSize=%u, ret=%d.\n",  dataSize, aclRet);
            free(data);
            return false;
        }
    } else {
        data = dataDev;
        std::cout << "is a device, don't cpoy to host" << std::endl;
    }

    FILE *outFileFp;

    if(flag){
        outFileFp = fopen(fileName, "wb+");
    }
    else{
        outFileFp = fopen(fileName, "ab+");
    }
    // outFileFp = fopen(fileName, "wb+");
    if (outFileFp == nullptr) {
        ATLAS_LOG_ERROR("fopen out file %s failed, error=%s.\n",  fileName, strerror(errno));
        free(data);
        return false;
    }

    bool ret = true;
    size_t writeRet = fwrite(data, 1, dataSize, outFileFp);
    if (writeRet != dataSize) {
        ATLAS_LOG_ERROR("need write %u bytes to %s, but only write %zu bytes, error=%s.\n",
        dataSize, fileName, writeRet, strerror(errno));
        ret = false;
    }

    if (!(RunStatus::GetDeviceStatus())) {
        free(data);
    }
    fflush(outFileFp);
    fclose(outFileFp);
    outFileFp = nullptr;
    return ret;
}

acldvppStreamFormat
h264_ffmpeg_profile_to_acl_stream_fromat(int profile) {
  switch (profile) {
  case 77: // h264 main level
    return H264_MAIN_LEVEL;
  case 66: // h264 baseline level
    return H264_BASELINE_LEVEL;
  case 100: // h264 high level
    return H264_HIGH_LEVEL;
  }
}

void* CopyDataToDevice(void* data, uint32_t dataSize, aclrtMemcpyKind policy) {
    void* buffer = nullptr;
    aclError aclRet = aclrtMalloc(&buffer, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("malloc device data buffer failed, aclRet is %d", aclRet);
        return nullptr;
    }
    // 实现Host内、Host与Device之间、Device内的同步内存复制
    // 系统内部会根据源内存地址指针、目的内存地址指针判断是否可以将源地址的数据复制到目的地址，如果不可以，则系统会返回报错
    aclRet = aclrtMemcpy(buffer, dataSize, data, dataSize, policy); 
    if (aclRet != ACL_ERROR_NONE) {
        ATLAS_LOG_ERROR("Copy data to device failed, aclRet is %d", aclRet);
        (void)aclrtFree(buffer);
        return nullptr;
    }

    return buffer;
}

void* CopyDataDeviceToDevice(void* deviceData, uint32_t dataSize) {
    return CopyDataToDevice(deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_DEVICE);     // Device内的内存复制
}

void* CopyDataHostToDevice(void* deviceData, uint32_t dataSize) {
    return CopyDataToDevice(deviceData, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);       // Host到Device的内存复制
}


char* GetPicDevBuffer4JpegD(const PicDesc &picDesc, uint32_t &devPicBufferSize)
{
    if (picDesc.picName.empty()) {
        ATLAS_LOG_ERROR("picture file name is empty");
        return nullptr;
    }

    uint32_t inputBuffSize = 0;
    void* inputBuff = ReadBinFile(picDesc.picName, inputBuffSize);
    if (inputBuff == nullptr) {
        ATLAS_LOG_ERROR("malloc inputHostBuff failed");
        return nullptr;
    }

    void *inBufferDev = nullptr;
    uint32_t inBufferSize = inputBuffSize;
    aclError aclRet;
    if (!(RunStatus::GetDeviceStatus())) {
        aclRet = acldvppMalloc(&inBufferDev, inBufferSize);
        if (aclRet != ACL_ERROR_NONE) {
            ATLAS_LOG_ERROR("malloc inBufferSize failed, aclRet is %d", aclRet);
            free(inputBuff);
            return nullptr;
        }

        aclRet = aclrtMemcpy(inBufferDev, inBufferSize, inputBuff, inputBuffSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_ERROR_NONE) {
            ATLAS_LOG_ERROR("memcpy from host to device failed. aclRet is %d", aclRet);
            acldvppFree(inBufferDev);
            free(inputBuff);
            return nullptr;
        }
        free(inputBuff);
    } else {
        inBufferDev = inputBuff;
    }

    devPicBufferSize = inBufferSize;
    return reinterpret_cast<char *>(inBufferDev);
}

void* ReadBinFile(std::string fileName, uint32_t &fileSize)
{
    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        ATLAS_LOG_ERROR("open file %s failed", fileName.c_str());
        return nullptr;
    }

    binFile.seekg(0, binFile.end);
    uint32_t binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        ATLAS_LOG_ERROR("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return nullptr;
    }

    binFile.seekg(0, binFile.beg);
    void* binFileBufferData = nullptr;
    if (!(RunStatus::GetDeviceStatus())) {
        binFileBufferData = malloc(binFileBufferLen);
        if (binFileBufferData == nullptr) {
            ATLAS_LOG_ERROR("malloc binFileBufferData failed");
            binFile.close();
            return nullptr;
        }
    } else {
        aclError aclRet = acldvppMalloc(&binFileBufferData, binFileBufferLen);
        if (aclRet !=  ACL_ERROR_NONE) {
            ATLAS_LOG_ERROR("malloc device data buffer failed, aclRet is %d", aclRet);
            return nullptr;
        }
    }

    binFile.read(static_cast<char *>(binFileBufferData), binFileBufferLen);
    binFile.close();
    fileSize = binFileBufferLen;
    return binFileBufferData;
}

/**
 * Convert the current time to the format "%Y%m%d%H%M%S"
 *
 * @param timeString buffer to save the time string with format "%Y%m%d%H%M%S"
 */
void GetCurTimeString(std::string &timeString)
{
    // Result file name use the time stamp as a suffix
    const int timeZoneDiff = 28800; // 8 hour time difference
    const int timeStringSize = 32;
    char timeStr[timeStringSize] = {0};
    time_t tmValue = time(nullptr) + timeZoneDiff;
    struct tm tmStruct = {0};
#ifdef _WIN32
    if (0 == gmtime_s(&tmStruct, &tmValue)) {
#else
    if (nullptr != gmtime_r(&tmValue, &tmStruct)) {
#endif
        strftime(timeStr, sizeof(timeStr), "%Y%m%d%H%M%S", &tmStruct);
    }
    timeString = timeStr;
    return;
}