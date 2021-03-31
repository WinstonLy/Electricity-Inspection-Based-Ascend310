/*
* @Author: winston
* @Date:   2021-01-07 17:08:57
* @Last Modified by:   WinstonLy
* @Last Modified time: 2021-03-30 15:54:39
* @Description: 
* @FilePath: /home/winston/AscendProjects/rtsp_dvpp_infer_dvpp_rtmp_test/atlas200dk_yolov4/Electricity-Inspection-Based-Ascend310/src/DvppVdec.cpp 
*/
#include <iostream>
#include <string>

#include "DvppVdec.h"
extern fstream resultVdec;

// 用户自定义数据 userdata
class VdecContext{
public:
    VdecContext(DvppVdec* _vdec, AVPacket* pkt): vdec(_vdec), packet(pkt){}
    DvppVdec* vdec;
    AVPacket* packet;
};

// 回调函数
static void VdecCallback(acldvppStreamDesc* input, acldvppPicDesc* output, void* userdata)
{
	// ATLAS_LOG_INFO("enter dvpp callback func");
    static int count = 1;
    VdecContext* ctx = (VdecContext*)userdata;
    CHECK_ACL(aclrtSetCurrentContext(*ctx->vdec->GetDeviceCtx()));


    if (output != nullptr) {
    	//获取VDEC解码的输出内存，调用自定义函数WriteToFile将输出内存中的数据写入文件后，再调用acldvppFree接口释放输出内存
        uint8_t* vdecOutBufferDev = (uint8_t*)acldvppGetPicDescData(output);
        if (vdecOutBufferDev != nullptr) {
            // 0: vdec success; others, vdec failed
            aclError retCode = acldvppGetPicDescRetCode(output);
            if (retCode == 0) {
                // process task: write file
                int size = acldvppGetPicDescSize(output);
                std::cout << "vdec callback output size: " << size << std::endl;
     //         // ATLAS_LOG_INFO("output size: %u", size);
                // std::string fileNameSave = "./data/dvpp_vdec" + std::to_string(count);
     //         // vdec输出结果在device侧，在WriteToFile方法中进行下述处理
 				// // 如果运行在host侧，则将device侧内存拷贝到host侧并保存；如果运行在device侧，则在device侧直接保存结果

                // bool flag = true;
                // if (!WriteToFile(fileNameSave.c_str(), vdecOutBufferDev, size, flag)) {
                //     ATLAS_LOG_ERROR("write file failed");
                // }


                // 继续resize图像
                ctx->vdec->GetHandler()(vdecOutBufferDev, size);
            } else {
                ATLAS_LOG_ERROR("vdec decode frame failed,err code = %d", retCode);
            }

            // 释放acldvppPicDesc类型的数据，表示解码后输出图片描述数据
            aclError ret = acldvppFree((void*)vdecOutBufferDev);
            if (ret != ACL_ERROR_NONE) {
                ATLAS_LOG_ERROR("fail to free output pic desc data ret = %d",  ret);
            }
        }
        // Destroy pic desc
        aclError ret = acldvppDestroyPicDesc(output);
        if (ret != ACL_ERROR_NONE) {
            ATLAS_LOG_ERROR("fail to Destroy output pic desc");
        }
    }

    

    av_packet_unref((AVPacket*)ctx->packet);
    delete ctx->packet;
    delete ctx;

    ++count; 

    // ATLAS_LOG_INFO("success to callback %d",  count);
}

DvppVdec::DvppVdec(): vdecHeight(0),vdecWidth(0), outputSize(0), 
    timeStamp(0), vdecChannelDesc(nullptr), format(PIXEL_FORMAT_YUV_SEMIPLANAR_420), enType(H264_HIGH_LEVEL)
{
}
DvppVdec::~DvppVdec(){

}
void DvppVdec::Destroy(){
	if (vdecChannelDesc != nullptr) {
        aclError ret = aclvdecDestroyChannel(vdecChannelDesc);
        if (ret != ACL_ERROR_NONE) {
            ATLAS_LOG_ERROR("acldvppDestroyChannel failed, ret = %d", ret);
        }
        aclvdecDestroyChannelDesc(vdecChannelDesc);
        vdecChannelDesc = nullptr;
    }
}

Result DvppVdec::Init(const pthread_t threadId, int height, 
	                int width, acldvppStreamFormat _enType, acldvppPixelFormat _format){
    timeStamp   = 0;
    vdecHeight  = ALIGN_UP2(height);
    vdecWidth   = ALIGN_UP16(width);
    outputSize  = (vdecHeight * vdecWidth * 3) / 2;
    enType      = _enType;
    format      = _format;

    // 创建视频流处理通道的通道描述信息
    vdecChannelDesc = aclvdecCreateChannelDesc();
    if (vdecChannelDesc == nullptr) {
        ATLAS_LOG_ERROR("fail to create vdec channel desc");
        return FAILED;
    }
    
    
    CHECK_ACL(aclvdecSetChannelDescChannelId(vdecChannelDesc, 10))
    CHECK_ACL(aclvdecSetChannelDescThreadId(vdecChannelDesc, threadId));
    CHECK_ACL(aclvdecSetChannelDescCallback(vdecChannelDesc, &VdecCallback));
    CHECK_ACL(aclvdecSetChannelDescEnType(vdecChannelDesc, enType));
    CHECK_ACL(aclvdecSetChannelDescOutPicFormat(vdecChannelDesc, format));
    // CHECK_ACL(aclvdecSetChannelDescOutPicWidth(vdecChannelDesc, width));
    // CHECK_ACL(aclvdecSetChannelDescOutPicHeight(vdecChannelDesc, height));

    // 是否实时出帧（发送一帧解码一帧，不依赖后续帧的传入）
    CHECK_ACL(aclvdecSetChannelDescOutMode(vdecChannelDesc, 0));
    aclError ret = aclvdecCreateChannel(vdecChannelDesc);
    if(ret != ACL_ERROR_NONE){
    	ATLAS_LOG_ERROR("create vdec channel failed, err code = %d", ret);
    	return FAILED;
    }


    // ATLAS_LOG_INFO("dvpp vdec init resource success");
    return SUCCESS;
}

Result DvppVdec::vdecSendFrame(AVPacket* pkt){
	// ATLAS_LOG_INFO("start send frame decoder");
    clock_t beginTime = clock();
	AVPacket *framePacket = new AVPacket();
    av_packet_ref(framePacket, pkt);
    // uint32_t outputSize = (modelWidth * modelHeight * 3) / 2;
    // uint32_t outputSize = (1520 * 2688 * 3) / 2;
    // 
  	// 创建输入视频码流描述信息，设置码流信息的属性
    acldvppStreamDesc *streamDesc = acldvppCreateStreamDesc();
    if (streamDesc == nullptr) {
        ATLAS_LOG_ERROR("fail to create input stream desc");
        return FAILED; 
    }
    // 要注意在设置描述信息时候的内存大小与实际图像大小是否一致
  	// framePacket->data 表示Device存放输入视频数据的内存，framePacket->size表示内存大小
    CHECK_ACL(acldvppSetStreamDescData(streamDesc, framePacket->data));
    CHECK_ACL(acldvppSetStreamDescSize(streamDesc, framePacket->size));
    CHECK_ACL(acldvppSetStreamDescFormat(streamDesc, aclvdecGetChannelDescEnType(vdecChannelDesc)));
    CHECK_ACL(acldvppSetStreamDescTimestamp(streamDesc, timeStamp));
    timeStamp++;
    
    

    // Malloc vdec output device memory
    void *picOutBufferDev{nullptr};
    CHECK_ACL(acldvppMalloc(&picOutBufferDev, outputSize))
    // uint8_t *picOutBufferDev = new uint8_t[outputSize];
    acldvppPicDesc *picOutputDesc = acldvppCreatePicDesc();

  	// 创建输出图片描述信息，设置图片描述信息的属性
    CHECK_ACL(acldvppSetPicDescData(picOutputDesc, picOutBufferDev));
    CHECK_ACL(acldvppSetPicDescSize(picOutputDesc, outputSize));
    // CHECK_ACL(acldvppSetPicDescWidth(picOutputDesc, vdecWidth));
    // CHECK_ACL(acldvppSetPicDescWidthStride(picOutputDesc, vdecWidth));
    // CHECK_ACL(acldvppSetPicDescHeight(picOutputDesc, vdecHeight));
    // CHECK_ACL(acldvppSetPicDescHeightStride(picOutputDesc, vdecHeight));
    CHECK_ACL(acldvppSetPicDescFormat(picOutputDesc, static_cast<acldvppPixelFormat>(format)));
	

  	// 执行视频码流解码，解码每帧数据后，系统自动调用callback回调函数将解码后的数据写入文件，再及时释放相关资源
    VdecContext* vCtx = new VdecContext(this, framePacket);
    clock_t b = clock();
    aclError ret = aclvdecSendFrame(vdecChannelDesc, streamDesc, picOutputDesc, nullptr, vCtx);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("vdec send frame failed, err code = %d", ret);
        return FAILED;
    }
    clock_t e = clock();
    // std::cout << "aclvdecSendFrame " << (double)(e - b)*1000/CLOCKS_PER_SEC << " ms" << std::endl;

    // if (ret != ACL_ERROR_NONE) {
    //     ATLAS_LOG_ERROR("fail to send frame, ret=%d", ret);
    //     if (framePacket != nullptr) {
    //         delete framePacket;
    //         framePacket = nullptr;
    //     }
    //     if (streamDesc != nullptr) {
    //         (void)acldvppDestroyStreamDesc(streamDesc);
    //         streamDesc = nullptr;
    //     }
    //     if (picOutBufferDev != nullptr) {
    //         (void)acldvppFree(picOutBufferDev);
    //         picOutBufferDev = nullptr;
    //     }
    //     if (picOutputDesc != nullptr) {
    //         (void)acldvppDestroyPicDesc(picOutputDesc);
    //         picOutputDesc = nullptr;
    //     }
    // }
    clock_t endTime = clock();
    resultVdec << "vdec a frame time: " << (double)(endTime - beginTime)*1000/CLOCKS_PER_SEC << " ms" <<endl;

    ATLAS_LOG_INFO("exit send frame decoder");
    return SUCCESS;
}

void DvppVdec::RegisterHandler(std::function<void(uint8_t *, size_t)> handler){
	vdecBufferHandler = handler;
}

const std::function<void(uint8_t *, size_t)> &DvppVdec::GetHandler() {
  return vdecBufferHandler;
}

void DvppVdec::SetDeviceCtx(aclrtContext *ctx){
    devCtx = ctx;
}
aclrtContext* DvppVdec::GetDeviceCtx(){
    return devCtx;
}
