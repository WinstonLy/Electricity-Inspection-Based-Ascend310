/*
* @Author: winston
* @Date:   2020-12-31 17:11:08
* @Last Modified by:   WinstonLy
* @Last Modified time: 2021-03-30 15:18:08
* @Description: 
* @FilePath: /home/winston/AscendProjects/rtsp_dvpp_infer_dvpp_rtmp_test/atlas200dk_yolov4/Electricity-Inspection-Based-Ascend310/src/DvppVpcResize.cpp 
*/
#include "DvppVpcResize.h"
#include <string>
extern fstream resultResize;
DvppVpcResize::DvppVpcResize(aclrtStream _stream):outputBufferSize(0), inputBufferSize(0),
    resizeInputBuffer(nullptr), resizeOutputBuffer(nullptr), resizeInputPicDesc(nullptr),
    resizeOutputPicDesc(nullptr), resizeConfig(nullptr), resizeChannelDesc(nullptr),stream(_stream)
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

	// create reszie config
	resizeConfig = acldvppCreateResizeConfig();
	// 设置缩放算法，目前之支持（华为自研的最近邻插值，0），（业界通用的Bilinear算法，1），（业界通用的Nearest neighbor算法，2）
    CHECK_ACL(acldvppSetResizeConfigInterpolation(resizeConfig, 0));

    // create channel description message
	resizeChannelDesc = acldvppCreateChannelDesc();
    // 创建resize通道
    CHECK_ACL(acldvppCreateChannel(resizeChannelDesc));


    // 创建resize 输入输出图片描述信息，并设置属性值
    resizeInputPicDesc = acldvppCreatePicDesc();
    resizeOutputPicDesc = acldvppCreatePicDesc();
    // 申请resize输入输出的内存空间，注意yuv格式的size计算,因为vdec的输出安16*2对齐
    inputBufferSize = yuv420sp_size(align_up(srcHeight, 2), align_up(srcWidth, 16));
    CHECK_ACL(acldvppMalloc(&resizeInputBuffer, inputBufferSize));
    
    outputBufferSize = yuv420sp_size(dstHeight, dstWidth);
    // resize的输出也要按照16*2对齐，但是模型输入如果不是对齐还要对齐吗？对齐之后可能不满足模型输入
    // outputBufferSize = yuv420sp_size(align_up(dstHeight, 2), align_up(dstWidth, 16));
    
    CHECK_ACL(acldvppMalloc(&resizeOutputBuffer, outputBufferSize)); 

    std::cout << "resize input buffer size: " << inputBufferSize 
              << " resize ouput buffer size: " << outputBufferSize <<std::endl;
    // 设置resize输入图片描述信息的属性值
    CHECK_ACL(acldvppSetPicDescData(resizeInputPicDesc, resizeInputBuffer));
    CHECK_ACL(acldvppSetPicDescFormat(resizeInputPicDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
    CHECK_ACL(acldvppSetPicDescWidth(resizeInputPicDesc, srcWidth));
    CHECK_ACL(acldvppSetPicDescHeight(resizeInputPicDesc, srcHeight));
    CHECK_ACL(acldvppSetPicDescWidthStride(resizeInputPicDesc, align_up(srcWidth, 16)));
    CHECK_ACL(acldvppSetPicDescHeightStride(resizeInputPicDesc, align_up(srcHeight, 2)));
    CHECK_ACL(acldvppSetPicDescSize(resizeInputPicDesc, inputBufferSize));

    // 设置resize输出图片描述信息的属性值
    CHECK_ACL(acldvppSetPicDescData(resizeOutputPicDesc, resizeOutputBuffer));
    CHECK_ACL(acldvppSetPicDescFormat(resizeOutputPicDesc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
    CHECK_ACL(acldvppSetPicDescWidth(resizeOutputPicDesc, dstWidth));
    CHECK_ACL(acldvppSetPicDescHeight(resizeOutputPicDesc, dstHeight));
    // 模型输入和VPC输出如果不满足16*2对齐
    // CHECK_ACL(acldvppSetPicDescWidthStride(resizeInputPicDesc, align_up(srcWidth, 16)));
    // CHECK_ACL(acldvppSetPicDescHeightStride(resizeInputPicDesc, align_up(srcHeight, 2)));
    CHECK_ACL(acldvppSetPicDescWidthStride(resizeOutputPicDesc, dstWidth));
    CHECK_ACL(acldvppSetPicDescHeightStride(resizeOutputPicDesc, dstHeight));
    CHECK_ACL(acldvppSetPicDescSize(resizeOutputPicDesc, outputBufferSize));

    ATLAS_LOG_INFO("dvpp vpc resize init success");
    return ACL_ERROR_NONE;
}


Result DvppVpcResize::Resize(void* pdata, size_t size){
    
    clock_t beginTime = clock();

	memcpy(resizeInputBuffer, pdata, size);
    ATLAS_LOG_INFO("resize input data size:%d", size);
    // resizeInputBuffer = pdata;
    // pdata = nullptr;
    // 设置原始图像的数据信息
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

    // 存储resize之后的图片
    // static int count = 0;
    // std::string fileNameSave = "./out/result/output_" + std::to_string(count) + ".yuv";
    // bool flag = false;
    // WriteToFile(fileNameSave.c_str(), resizeOutputBuffer, outputBufferSize, flag);
    // ++count;
    // 
    // 存储resize之前的数据
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




   