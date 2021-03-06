/*
* @Author: winston
* @Date:   2020-12-31 17:11:08
* @Last Modified by:   WinstonLy
* @Last Modified time: 2021-04-13 22:00:51
* @Description: 
* @FilePath: /home/winston/AscendProjects/rtsp_dvpp_infer_dvpp_rtmp_test/atlas200dk_yolov4/Electricity-Inspection-Based-Ascend310/src/dvpp_vpc.cpp 
*/
#include "dvpp_vpc.h"
#include <string>
#include "model_infer.h"
#include <queue>
#include <mutex>
#include <chrono>
#include <thread>

extern fstream resultResize;


namespace {
  int modelWidth = 608;
  int modelHeight = 608;
  const char* modelPath = "./model/yolov4.om";
  const char* appConf = "./script/object_detection.conf";
}

DvppVpcResize::DvppVpcResize(aclrtStream _stream):outputBufferSize(0), inputBufferSize(0),
    resizeInputBuffer(nullptr), resizeOutputBuffer(nullptr), resizeInputPicDesc(nullptr),
    resizeOutputPicDesc(nullptr), resizeConfig(nullptr), resizeChannelDesc(nullptr),
    stream(_stream),_srcWidth(0), _srcHeight(0), _dstWidth(0), _dstHeight(0)
{
    // srcData.width = 0;
    // srcData.height = 0;
    // srcData.alignWidth = 16;
    // srcData.alignHeight = 2;
    // srcData.size = 0;
    // srcData.data = nullptr;
}
DvppVpcResize::~DvppVpcResize(){
   
}
void DvppVpcResize::Destroy(){
	if(resizeChannelDesc != nullptr){
		aclError ret = acldvppDestroyChannel(resizeChannelDesc);
		if(ret != ACL_ERROR_NONE){
			ATLAS_LOG_ERROR("Destroy resize channel desc failed, err code = %d", ret);
		}
		acldvppDestroyChannelDesc(resizeChannelDesc);
		resizeChannelDesc = nullptr;
	}

	if(resizeConfig != nullptr){
		acldvppDestroyResizeConfig(resizeConfig);
        resizeConfig = nullptr;
	}

	if(resizeInputPicDesc != nullptr){
		acldvppDestroyPicDesc(resizeInputPicDesc);
		resizeInputPicDesc = nullptr;
	}

	if(resizeOutputPicDesc != nullptr){
		acldvppDestroyPicDesc(resizeOutputPicDesc);
		resizeOutputPicDesc = nullptr;
	}

	if(resizeInputBuffer != nullptr){
		acldvppFree(resizeInputBuffer);
		resizeInputBuffer = nullptr;
	}

	if(resizeOutputBuffer != nullptr){
		acldvppFree(resizeOutputBuffer);
		resizeOutputBuffer = nullptr;
	}


    ATLAS_LOG_INFO("DvppVpcResize::~DvppVpcResize End");
}

aclError DvppVpcResize::Init(int srcWidth, int srcHeight, int dstWidth, int dstHeight){

    _srcWidth  = srcWidth;
    _srcHeight = srcHeight;
    _dstWidth  = dstWidth;
    _dstHeight = dstHeight;

	// create reszie config
	resizeConfig = acldvppCreateResizeConfig();
	// ????????????????????????????????????????????????????????????????????????0????????????????????????Bilinear?????????1????????????????????????Nearest neighbor?????????2???
    CHECK_ACL(acldvppSetResizeConfigInterpolation(resizeConfig, 0));

    // create channel description message
	resizeChannelDesc = acldvppCreateChannelDesc();
    // ??????resize??????
    CHECK_ACL(acldvppCreateChannel(resizeChannelDesc));


    // ??????resize ???????????????????????????????????????????????????
    resizeInputPicDesc = acldvppCreatePicDesc();
    resizeOutputPicDesc = acldvppCreatePicDesc();
    // ??????resize????????????????????????????????????yuv?????????size??????,??????vdec????????????16*2??????
    inputBufferSize = yuv420sp_size(align_up(_srcHeight, 2), align_up(_srcWidth, 16));
    CHECK_ACL(acldvppMalloc(&resizeInputBuffer, inputBufferSize));
    
    outputBufferSize = yuv420sp_size(_dstHeight, _dstWidth);
    // resize?????????????????????16*2??????????????????????????????????????????????????????????????????????????????????????????????????????
    // outputBufferSize = yuv420sp_size(align_up(dstHeight, 2), align_up(dstWidth, 16));
    
    CHECK_ACL(acldvppMalloc(&resizeOutputBuffer, outputBufferSize)); 

    // std::cout << "resize input buffer size: " << inputBufferSize 
              // << " resize ouput buffer size: " << outputBufferSize <<std::endl;
    // ??????resize????????????????????????????????????
    CHECK_ACL(acldvppSetPicDescData(resizeInputPicDesc, resizeInputBuffer));
    CHECK_ACL(acldvppSetPicDescFormat(resizeInputPicDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
    CHECK_ACL(acldvppSetPicDescWidth(resizeInputPicDesc, _srcWidth));
    CHECK_ACL(acldvppSetPicDescHeight(resizeInputPicDesc, _srcHeight));
    CHECK_ACL(acldvppSetPicDescWidthStride(resizeInputPicDesc, align_up(_srcWidth, 16)));
    CHECK_ACL(acldvppSetPicDescHeightStride(resizeInputPicDesc, align_up(_srcHeight, 2)));
    CHECK_ACL(acldvppSetPicDescSize(resizeInputPicDesc, inputBufferSize));

    // ??????resize????????????????????????????????????
    CHECK_ACL(acldvppSetPicDescData(resizeOutputPicDesc, resizeOutputBuffer));
    CHECK_ACL(acldvppSetPicDescFormat(resizeOutputPicDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
    CHECK_ACL(acldvppSetPicDescWidth(resizeOutputPicDesc, _dstWidth));
    CHECK_ACL(acldvppSetPicDescHeight(resizeOutputPicDesc, _dstHeight));
    // ???????????????VPC?????????????????????16*2??????
    // CHECK_ACL(acldvppSetPicDescWidthStride(resizeInputPicDesc, align_up(srcWidth, 16)));
    // CHECK_ACL(acldvppSetPicDescHeightStride(resizeInputPicDesc, align_up(srcHeight, 2)));
    CHECK_ACL(acldvppSetPicDescWidthStride(resizeOutputPicDesc, _dstWidth));
    CHECK_ACL(acldvppSetPicDescHeightStride(resizeOutputPicDesc, _dstHeight));
    CHECK_ACL(acldvppSetPicDescSize(resizeOutputPicDesc, outputBufferSize));

    ATLAS_LOG_INFO("dvpp vpc resize init success");
    
    return ACL_ERROR_NONE;
}


Result DvppVpcResize::Resize(void* pdata, size_t size){
    
    clock_t beginTime = clock();

	memcpy(resizeInputBuffer, pdata, size);

    // ATLAS_LOG_INFO("resize input data size:%d", size);
    // resizeInputBuffer = pdata;
    // pdata = nullptr;
    // ?????????????????????????????????
    // if(size == srcData.size){
    //     ATLAS_LOG_INFO("resize input size == vdec output size");
    // }
    // srcData.data = resizeInputBuffer;

	aclError ret = acldvppVpcResizeAsync(resizeChannelDesc, resizeInputPicDesc,
	                                     resizeOutputPicDesc, resizeConfig, stream);
    if(ret != ACL_ERROR_NONE){
    	ATLAS_LOG_ERROR("dvpp resize failed, err code = %d", ret);
    	return FAILED;
    }
    ret = aclrtSynchronizeStream(stream);
    if(ret != ACL_ERROR_NONE){
    	ATLAS_LOG_ERROR("dvpp resize synchronize strem failed, err code = %d", ret);
    	return FAILED;
    }

    // ??????resize???????????????
    // static int count = 0;
    // std::string fileNameSave = "./out/result/output_" + std::to_string(count) + ".yuv";
    // bool flag = false;
    // WriteToFile(fileNameSave.c_str(), resizeOutputBuffer, outputBufferSize, flag);
    // ++count;
    // 
    // ??????resize???????????????
    // static int countImage = 0;
    // std::string file = "./out/result/input_" + std::to_string(countImage) + ".yuv";
    // // bool flag = false;
    // WriteToFile(file.c_str(), resizeInputBuffer, inputBufferSize, flag);
    // ++countImage;
    
  

    clock_t endTime = clock();
    resultResize << "resize a frame time: " << (double)(endTime - beginTime)*1000/CLOCKS_PER_SEC << " ms" <<endl;


    if(bufferHandler){
        bufferHandler((uint8_t*)resizeOutputBuffer);
    }
    

    
    ATLAS_LOG_INFO("dvpp resize end");
    return SUCCESS;
}

void DvppVpcResize::RegisterHandler(std::function<void(uint8_t *)> handler){
	bufferHandler = handler;
}

uint8_t *DvppVpcResize::GetOutputBuffer(){

	return (uint8_t*)resizeOutputBuffer;
}

int DvppVpcResize::GetOutputBufferSize(){

    return outputBufferSize;
}

// void DvppVpcResize::GetSrcData(ImageData& frameData){
//     frameData = srcData;
//     if(frameData.data == nullptr){
//         ATLAS_LOG_ERROR("get src data failed");
//     }
//     ATLAS_LOG_INFO("srcData size = %d, frameData size = %d", srcData.size, frameData.size);
//     ATLAS_LOG_INFO("frameData info, width = %d, height = %d, size = %d",
//                              frameData.width, frameData.height, frameData.size);
// }

uint8_t* DvppVpcResize::GetInputBuffer(){

    return (uint8_t*)resizeInputBuffer;

}




   