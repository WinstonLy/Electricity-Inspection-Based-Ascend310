/*
* @Author: winston
* @Date:   2021-01-07 09:24:43
* @Last Modified by:   WinstonLy
* @Last Modified time: 2021-04-02 20:45:53
* @Description: 
* @FilePath: /home/winston/AscendProjects/rtsp_dvpp_infer_dvpp_rtmp_test/atlas200dk_yolov4/Electricity-Inspection-Based-Ascend310/src/dvpp_venc.cpp 
*/
#include <iostream>

#include "dvpp_venc.h"
#include "ascenddk/presenter/agent/presenter_channel.h"
// send venc frame 时候的自定义数据
class VencContext{
public:
    VencContext(DvppVenc* _venc, uint8_t* _frame, vector<DetectionResult>& _results):venc(_venc), 
                frame(_frame), results(_results)
                {}
    VencContext(DvppVenc* _venc, uint8_t* _frame,size_t _size):venc(_venc), 
                frame(_frame), size(_size)
                {}
    DvppVenc *venc;
    uint8_t* frame;
    size_t size;
    vector<DetectionResult> results;
};


static void VencCallBack(acldvppPicDesc* input, acldvppStreamDesc* output, void* userdata){
	ATLAS_LOG_INFO("enter into venc callback");

	static int count = 0;

    VencContext* vCtx = (VencContext*)userdata;
	if(output != nullptr){
		// 获取视频编码结果数据，并写入文件
		uint8_t* vencOutputDev = (uint8_t*)acldvppGetStreamDescData(output);

		if(vencOutputDev != nullptr){
            // 0: venc success, other:failed
            aclError ret = acldvppGetStreamDescRetCode(output);

            if(ret == 0){
            	uint32_t size = acldvppGetStreamDescSize(output);
            	ATLAS_LOG_INFO("venc buffer size = %d", size);

            	std::string fileNameSave = "./data/dvpp_venc.h264";
            	bool flag = false;

            	if(!WriteToFile(fileNameSave.c_str(), vencOutputDev, size, flag)){
            		ATLAS_LOG_ERROR("save venc data daile");
            	}

                // ATLAS_LOG_INFO("start send venc frame to display");
                // vCtx->venc->GetHandler()(vencOutputDev, size);
                // vCtx->venc->SendImageDisplay(vCtx->results, vencOutputDev, size);
            }
            else{
            	ATLAS_LOG_ERROR("venc frame failed, err code=%d", ret);
            }  
		}

        
	}

    delete vCtx;

	ATLAS_LOG_INFO("exit venc callback %d", count++);	
}
DvppVenc::DvppVenc(aclrtStream& _stream):stream(_stream),
    vencChannelDesc(nullptr), vencFrameConfig(nullptr),
    vencInputPicDesc(nullptr), vencHeight(0), vencWidth(0), 
    isDevice(true), vencSize(0), channel(nullptr)
{
	
}
DvppVenc::~DvppVenc(){

}
void DvppVenc::Destroy(){
	aclvencDestroyChannel(vencChannelDesc);
    aclvencDestroyChannelDesc(vencChannelDesc);
    acldvppDestroyPicDesc(vencInputPicDesc);
    aclvencDestroyFrameConfig(vencFrameConfig);

    delete channel;

    ATLAS_LOG_INFO("DvppVenc::~DvppVenc End");
}
Result DvppVenc::Init(const pthread_t pthreadId, int height, int width){
    vencHeight = align_up(height, 2);
    vencWidth  = align_up(width, 16);   
    vencSize   = (vencHeight * vencWidth * 3) / 2;
    std::cout << "DvppVenc::Init height: " << vencHeight << " width " << vencWidth << " size " << vencSize << std::endl;

 
    // 创建venc通道描述信息
	vencChannelDesc = aclvencCreateChannelDesc();
	if(vencChannelDesc == nullptr){
		aclvencDestroyChannelDesc(vencChannelDesc);
	}
	// 创建venc帧配置文件
	vencFrameConfig = aclvencCreateFrameConfig();
	if(vencFrameConfig == nullptr){
		aclvencDestroyFrameConfig(vencFrameConfig);
	}
    // 创建venc输入图片描述信息
	vencInputPicDesc = acldvppCreatePicDesc();

    // 创建编码通道描述属性
    CHECK_ACL(aclvencSetChannelDescThreadId(vencChannelDesc, pthreadId));
    CHECK_ACL(aclvencSetChannelDescCallback(vencChannelDesc, VencCallBack));
    CHECK_ACL(aclvencSetChannelDescEnType(vencChannelDesc, H264_HIGH_LEVEL));
    CHECK_ACL(aclvencSetChannelDescPicFormat(vencChannelDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
    CHECK_ACL(aclvencSetChannelDescPicHeight(vencChannelDesc, vencHeight));
    CHECK_ACL(aclvencSetChannelDescPicWidth(vencChannelDesc, vencWidth));
    CHECK_ACL(aclvencSetChannelDescKeyFrameInterval(vencChannelDesc, 1)); // 关键帧间隔

    // 创建编码通道
    CHECK_ACL(aclvencCreateChannel(vencChannelDesc));

    // 创建单帧编码配置数据并设置，不是结束帧
    CHECK_ACL(aclvencSetFrameConfigForceIFrame(vencFrameConfig, 0));
    CHECK_ACL(aclvencSetFrameConfigEos(vencFrameConfig, 0));

    //Connect the presenter server
    Result ret = OpenPresenterChannel();
    if (ret != SUCCESS) {
        ATLAS_LOG_ERROR("Open presenter channel failed");
        return FAILED;
    }

    return SUCCESS;
}   

aclError DvppVenc::SendVencFrame(vector<DetectionResult>& modelResults, uint8_t* frameData){
    // 设置图片描述信息
    ATLAS_LOG_INFO("start send venc frame");
	CHECK_ACL(acldvppSetPicDescData(vencInputPicDesc, frameData));
	CHECK_ACL(acldvppSetPicDescSize(vencInputPicDesc, vencSize));

	CHECK_ACL(acldvppSetPicDescFormat(vencInputPicDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
    CHECK_ACL(acldvppSetPicDescWidth(vencInputPicDesc, vencWidth));
    CHECK_ACL(acldvppSetPicDescHeight(vencInputPicDesc, vencHeight));
    CHECK_ACL(acldvppSetPicDescWidthStride(vencInputPicDesc, vencWidth));
    CHECK_ACL(acldvppSetPicDescHeightStride(vencInputPicDesc, vencHeight));


    VencContext* vCtx = new VencContext(this, frameData, modelResults);
    aclError ret = aclvencSendFrame(vencChannelDesc, vencInputPicDesc, nullptr, 
                                    vencFrameConfig, vCtx);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("send venc frame failed, err code = %d", ret);
        return ret;
    }

    // // 设置单帧编码配置数据，是结束帧
    // CHECK_ACL(aclvencSetFrameConfigForceIFrame(vencFrameConfig, 1));
    // CHECK_ACL(aclvencSetFrameConfigEos(vencFrameConfig, 0));
    
    // // 执行最后一帧视频编码
    // CHECK_ACL(aclvencSendFrame(vencChannelDesc, nullptr, nullptr, vencFrameConfig, nullptr));

    return ACL_ERROR_NONE;
}


aclError DvppVenc::SendVencFrame(uint8_t* frameData, size_t size){
    // 设置图片描述信息
    ATLAS_LOG_INFO("start send venc frame");
    CHECK_ACL(acldvppSetPicDescData(vencInputPicDesc, frameData));
    CHECK_ACL(acldvppSetPicDescSize(vencInputPicDesc, vencSize));

    CHECK_ACL(acldvppSetPicDescFormat(vencInputPicDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
    CHECK_ACL(acldvppSetPicDescWidth(vencInputPicDesc, vencWidth));
    CHECK_ACL(acldvppSetPicDescHeight(vencInputPicDesc, vencHeight));
    CHECK_ACL(acldvppSetPicDescWidthStride(vencInputPicDesc, vencWidth));
    CHECK_ACL(acldvppSetPicDescHeightStride(vencInputPicDesc, vencHeight));


    VencContext* vCtx = new VencContext(this, frameData, size);
    aclError ret = aclvencSendFrame(vencChannelDesc, vencInputPicDesc, nullptr, 
                                    vencFrameConfig, vCtx);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("send venc frame failed, err code = %d", ret);
        return ret;
    }

    // // 设置单帧编码配置数据，是结束帧
    // CHECK_ACL(aclvencSetFrameConfigForceIFrame(vencFrameConfig, 1));
    // CHECK_ACL(aclvencSetFrameConfigEos(vencFrameConfig, 0));
    
    // // 执行最后一帧视频编码
    // CHECK_ACL(aclvencSendFrame(vencChannelDesc, nullptr, nullptr, vencFrameConfig, nullptr));

    return ACL_ERROR_NONE;
}


void DvppVenc::RegisterHandler(std::function<void(uint8_t *, int)> handler){
    vencBufferHandler = handler;
}

const std::function<void(uint8_t *, int)> &DvppVenc::GetHandler(){
    return vencBufferHandler;
}

int DvppVenc::GetVencWidth(){
    return vencWidth;
}
int DvppVenc::GetVencHeight(){
    return vencHeight;
}
Result DvppVenc::OpenPresenterChannel() {
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

Result DvppVenc::SendImageDisplay(vector<DetectionResult>& detectionResults,
                        uint8_t* data, int size) {

    ATLAS_LOG_INFO("start send image dispaly");
    ImageFrame imageParam;
    imageParam.format = ImageFormat::kJpeg;
    imageParam.width = vencWidth;
    imageParam.height = vencHeight;
    imageParam.size = size;
    imageParam.data = reinterpret_cast<uint8_t*>(data);
    imageParam.detection_results = detectionResults;
    //Sends the detected object frame information and frame image to the Presenter Server for display
    PresenterErrorCode errorCode = PresentImage(channel, imageParam);
    if (errorCode != PresenterErrorCode::kNone) {
        ATLAS_LOG_ERROR("PresentImage failed %d", static_cast<int>(errorCode));
        return FAILED;
    }

    return SUCCESS;
}


