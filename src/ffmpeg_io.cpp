/*
* @Author: winston
* @Date:   2021-01-09 16:06:22
* @Last Modified by:   WinstonLy
* @Last Modified time: 2021-04-10 23:31:49
* @Description: 
* @FilePath: /home/winston/AscendProjects/rtsp_dvpp_infer_dvpp_rtmp_test/atlas200dk_yolov4/Electricity-Inspection-Based-Ascend310/src/ffmpeg_io.cpp 
*/
#include "ffmpeg_io.h"

#include <queue>
#include <mutex>
#include <chrono>
#include <thread>

extern bool runFlag;
extern std::mutex mtxQueueRtsp;
extern std::queue<std::pair<int, AVPacket> > queueRtsp;
extern fstream resultFFmpeg;

// FFMPEGInput 成员函数
FFMPEGInput::FFMPEGInput():avfcRtspInput(nullptr), bsfc(nullptr),
    avcpRtspInput(nullptr), decoderContext(nullptr), bsfFilter(nullptr), 
    needBsf(false), videoIndex(-1)
{

}
FFMPEGInput::~FFMPEGInput(){
	  avformat_close_input(&avfcRtspInput);
    avformat_free_context(avfcRtspInput);
}

void FFMPEGInput::InputInit(const char* inputPath){
    clock_t beginTime = clock();
    // ffmpeg init
    av_register_all();						// Initialize libavformat and register all the muxers(复用器）, demuxers（解复用器） and protocols.
    avformat_network_init();				// Do global initialization of network components.
    // av_log_set_level(AV_LOG_DEBUG);			// Set the log level

    // init ffmpeg input
    // 申请输入rtsp
    ATLAS_LOG_INFO("[rtsp input]: %s",inputPath);
    
    // 初始化 rtsp input AVFormatContext 对象
    // AVFormatContext: 描述了一个媒体文件或媒体流的构成和基本信息
    avfcRtspInput = avformat_alloc_context();	// Allocate an AVFormatContext.

    // 设定探测相关内部参数
    AVDictionary *avdic{nullptr};	
    // 设置为 tcp 传输，最大延时时间
    av_dict_set(&avdic, "rtsp_transport", "tcp", 0);
    av_dict_set(&avdic, "max_dealy", "100000000", 0);	
    av_dict_set(&avdic, "buffer_size", "10485760", 0); 
    av_dict_set(&avdic, "stimeout", "5000000", 0);
    av_dict_set(&avdic, "pkt_size", "10485760", 0); 
    av_dict_set(&avdic, "reorder_queue_size", "0", 0);

    // Open an input stream and read the header. The codecs are not opened.
    uint8_t ret = avformat_open_input(&avfcRtspInput, inputPath, nullptr, &avdic);
    if(ret != 0){
    	  ATLAS_LOG_ERROR(" can't open input: %s, ffmpeg err code: %u", inputPath, ret);
    	  return;
    }

    // Read packets of a media file to get stream information.
    ret = avformat_find_stream_info(avfcRtspInput, nullptr);
    if(ret != 0){
    	  ATLAS_LOG_ERROR(" can't find stream, ffmpeg err code: %u", ret);
    	  return;
    }
    // print rtsp input message,第四个参数为0表示输入，为1表示输出
    av_dump_format(avfcRtspInput, 0, inputPath, 0);

    // 查找码流中是否有视频流
    for(uint8_t i = 0; i < avfcRtspInput->nb_streams; ++i){
    	  if (avfcRtspInput->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
    		    videoIndex = i;
    		    break;
    	  }
    }
    // videoIndex = av_find_best_stream(avfcRtspInput, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (videoIndex < 0) {
    	  ATLAS_LOG_ERROR(" failed to find a video stream AVMEDIA_TYPE_VIDEO");
    	  return;
    }
    ATLAS_LOG_INFO("success to find a video stream");

    // print video frame info
    // AVCodecParameters: This struct describes the properties of an encoded stream
    avcpRtspInput = avfcRtspInput->streams[videoIndex]->codecpar;

    cout << "[FFMPEGInput::Init] " << inputPath << " codec name:" << avcodec_get_name(avcpRtspInput->codec_id) << endl;
    cout << "avcc profile: " << avcpRtspInput->profile << endl;
    cout << "frame h: " << avcpRtspInput->height << " frame w: " << avcpRtspInput->width << endl;
    cout << "codec_tag " << avcpRtspInput->codec_tag << endl;
    cout << "extra_data size: " << avcpRtspInput->extradata_size << endl;

    // 判断是否编码格式为AVC1，如果是转换成h264，avc1不带起始码0x00000001

  	if(true || avcpRtspInput->codec_tag == AVC1_TAG){
  		// a bitstream filter with the specified name or NULL if no such bitstream filter exists.
	 	bsfFilter = av_bsf_get_by_name("h264_mp4toannexb");
  		
  	    ret = av_bsf_alloc(bsfFilter, &bsfc);
  	    if(ret < 0){
  			    ATLAS_LOG_ERROR(" failed to init bfsc, err code %u",   ret);
  			    return;
  	    }

  		 
  	    avcodec_parameters_copy(bsfc->par_in, avfcRtspInput->streams[videoIndex]->codecpar);
   	    // Prepare the filter for use, after all the parameters and options have been set.
       	av_bsf_init(bsfc);
      	needBsf = true;
  	}
	
  	
    clock_t endTime = clock();
    resultFFmpeg << "FFMpeg init time:" << (double)(endTime - beginTime)*1000/CLOCKS_PER_SEC << " ms" <<endl;
    ATLAS_LOG_INFO("ffmpeg rtsp input init success");

}
void FFMPEGInput::Run(){
    // 开始从rtsp流获取视频帧
    static int indexFrame = 0;
    fstream outfile;
    //outfile.open("./data/result.txt");
   
    AVPacket packet;
    av_init_packet(&packet);
    

    // need stop flag?
    runFlag = true;
    while(runFlag){
        clock_t beginTime = clock();
   
        // 从 avfcRtspInput 中读取码流进入 packet
        // clock_t beginTime = clock();
        uint32_t ret = av_read_frame(avfcRtspInput, &packet);
        // if(packet.data == nullptr)
        // {
        //     runFlag = false;
        //     break;
        // }
	    if (ret < 0) {
        	char err_buf[AV_ERROR_MAX_STRING_SIZE] = {0};
        	std::cerr << "[FFMPEGInput::ReceiveSinglePacket] err string: "
                 	  << av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, ret)
           		  << std::endl;
        	av_packet_unref(&packet);
        	ATLAS_LOG_ERROR("ffmpeg exit, get frame packet failed");
        	continue;
  	    }

        clock_t endTime = clock();
        resultFFmpeg << "FFmpeg read a frame time: " << (double)(endTime - beginTime)*1000/CLOCKS_PER_SEC << " ms" <<endl;
        
        // if(packetHandler && packet.stream_index == videoIndex){
        //     packetHandler(&packet);
        // }
        // else{
        //     // std::cout << "stream_index != videoIndex" << std::endl;
        //     continue;
        // }
	    
        if (packet.stream_index == videoIndex) {
            // test read frame packet->send vdec
            // send video packet to ffmeg
            ret = av_bsf_send_packet(bsfc, &packet);
            if (ret < 0) {
                std::cout << "av_bsf_send_packet failed" << std::endl;
                continue;
            }

		    // read a single frame from ffmpeg
            // ATLAS_LOG_INFO("decoder send freame");
            // packet_handler = DecoderSendFrame;
            // while (av_bsf_receive_packet(bsfc, &packet) == 0) {
            //     // 执行解码
            //     // if(packetHandler){
            //     // 	packetHandler(&packet);
            //     // }
            //     // av_packet_unref(&packet);
            //     
                
            // }
            
            // packet 进入队列
            if(av_bsf_receive_packet(bsfc, &packet) == 0)
            {
                mtxQueueRtsp.lock();
                queueRtsp.push(std::make_pair(indexFrame++, packet));
            }
            if(queueRtsp.size() >= 100){
                
                std::cout << "[WARNING] rtsp input size is " << queueRtsp.size() <<std::endl; 
                
                // 清空队列
                std::queue<std::pair<int, AVPacket> > tempQueue;
                swap(tempQueue, queueRtsp);
                mtxQueueRtsp.unlock();
            }
            else{
                mtxQueueRtsp.unlock();
            }
        }

        av_packet_unref(&packet);
        // --count;
        
        // clock_t endTime = clock();
        // resultFFmpeg << "FFmpeg read a frame time: " << (double)(endTime - beginTime)*1000/CLOCKS_PER_SEC << " ms" <<endl;
        // std::cout << "ffmepg read frame " << count << std::endl;
    }

    av_bsf_free(&bsfc);

    ATLAS_LOG_INFO("process end");
}
void FFMPEGInput::Destroy(){
    avformat_close_input(&avfcRtspInput);
    avformat_free_context(avfcRtspInput);
}
void FFMPEGInput::RegisterHandler(std::function<void(AVPacket *)> handler){
	packetHandler = handler;
}

int FFMPEGInput::GetWidth(){
	if (avcpRtspInput == nullptr) {
        throw std::runtime_error("FFMPEGInput Stream is not Inited!");
    }
    return avcpRtspInput->width;
}



int FFMPEGInput::GetHeight(){
	if (avcpRtspInput == nullptr) {
        throw std::runtime_error("FFMPEGInput Stream is not Inited!");
    }
    return avcpRtspInput->height;
}


acldvppStreamFormat FFMPEGInput::GetProfile(){
    if (avcpRtspInput == nullptr) {
        throw std::runtime_error("FFMPEGInput Stream is not Inited!");
    }
    return h264_ffmpeg_profile_to_acl_stream_fromat(avcpRtspInput->profile);
}
int FFMPEGInput::GetFrame(){
    if (avcpRtspInput == nullptr) {
        throw std::runtime_error("FFMPEGInput Stream is not Inited!");
    }
    return avfcRtspInput->streams[videoIndex]->avg_frame_rate.num / avfcRtspInput->streams[videoIndex]->avg_frame_rate.den;
    // return decoderContext->time_base.num / decoderContext->time_base.den;
}

//*******************FFMPEGOutput成员函数************************//
FFMPEGOutput::FFMPEGOutput(): height(0), width(0), streamName(""), isValid(false),
    codecOption(nullptr), videoFrame(nullptr), avccRtmpOutput(nullptr),
    avfcRtmpOutput(nullptr), rtmpStream(nullptr), videoCodec(nullptr)
{

}
FFMPEGOutput::~FFMPEGOutput(){
    Destroy();
}
void FFMPEGOutput::Init(int _height, int _width, int frameRate, int picFormat, std::string _name){
    height          = _height;
    width           = _width;
    AVRational avFrameRate;
    avFrameRate.num = frameRate;
    avFrameRate.den = 1;
    streamName      = _name;
    const char* rtmpOutputPath = streamName.c_str();    
    const char* profile        = "high444";
    std::string format         = "flv";

    // 初始化ffmpeg
    av_register_all();
    avformat_network_init();
    // av_log_set_level(AV_LOG_TRACE);
    // 
    // 初始化一个rtmp输出AVFormatContext结构体（一个媒体文件或媒体流的构成和基本信息）
    // 第一个参数是结构体，第二个参数确定输出格式，为NULL表示可以设定后两个参数
    // 第三个参数制定输出格式，第四个参数指定输出文件名
    int ret = avformat_alloc_output_context2(
        &avfcRtmpOutput, NULL, format.empty() ? NULL : format.c_str(), rtmpOutputPath);
    if(ret < 0){
        ATLAS_LOG_ERROR("[FFMPEGOutput::Init] avformat_alloc_output_context2 failed, err code = %d", ret);
        return;
    }

    // 设置修改（解）复用器行为的标志
    // AVFMT_FLAG_NOBUFFER:尽可能不缓存帧
    // AVFMT_FLAG_FLUSH_PACKETS：每个数据包都刷新AVIOContext 
    
    // encoder_avfc->flags |= AVFMT_FLAG_NOBUFFER;
    avfcRtmpOutput->flags |= AVFMT_FLAG_FLUSH_PACKETS;
    if (!(avfcRtmpOutput->oformat->flags & AVFMT_NOFILE)) {
        // 创建并且初始化用path访问资源的AVIOContext结构体，打开FFMpeg的输出文件，avfcRtmpOutput->pb:I/O context
        ret = avio_open2(&avfcRtmpOutput->pb, rtmpOutputPath, AVIO_FLAG_WRITE, NULL, NULL);
        if (ret < 0) {
            char err_buf[AV_ERROR_MAX_STRING_SIZE] = {0};
            std::cerr << "[FFMPEGOutput::Init] avio_open2 failed err code: " << ret
                << " Reason: "
                << av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, ret)
                << std::endl;
            return;
        }
    }

    // 查找具有匹配编解码器ID的注册编码器
    videoCodec = avcodec_find_encoder(AV_CODEC_ID_H264);

    // 初始化视频编解码器，和编解码器ID
    avfcRtmpOutput->video_codec = videoCodec;
    avfcRtmpOutput->video_codec_id = AV_CODEC_ID_H264;

    // 初始化一个AVCodecContext结构体（编解码结构体）
    avccRtmpOutput = avcodec_alloc_context3(videoCodec);

    // 初始化相关参数
    avccRtmpOutput->codec_tag  = 0;
    avccRtmpOutput->codec_id   = AV_CODEC_ID_H264;
    avccRtmpOutput->codec_type = AVMEDIA_TYPE_VIDEO;
    avccRtmpOutput->gop_size   = 12;    // 一组图片的图片数量
    avccRtmpOutput->height     = height;
    avccRtmpOutput->width      = width;
    avccRtmpOutput->pix_fmt    = (AVPixelFormat)picFormat; // AV_PIX_FMT_NV12;// NV12 IS YUV420
    // control rate
    avccRtmpOutput->bit_rate       = 2 * 1000 * 1000;  // 平均比特率
    avccRtmpOutput->rc_buffer_size = 4 * 1000 * 1000;  //解码器比特流缓冲区大小
    avccRtmpOutput->rc_max_rate    = 2 * 1000 * 1000;
    avccRtmpOutput->rc_min_rate    = 2.5 * 1000 * 1000;
    avccRtmpOutput->time_base.num  = avFrameRate.den;  // 基本时间，帧时间戳
    avccRtmpOutput->time_base.den  = avFrameRate.num;


    if (avfcRtmpOutput->oformat->flags & AVFMT_GLOBALHEADER) {
        avccRtmpOutput->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // 添加新的流到媒体文件中
    rtmpStream = avformat_new_stream(avfcRtmpOutput, videoCodec);

    // 从提供的编解码结构体填充相关的参数
    ret = avcodec_parameters_from_context(rtmpStream->codecpar, avccRtmpOutput);
    if (ret < 0) {
        std::cerr << "[FFMPEGOutput::Init] avcodec_parameters_from_context failed"
                << std::endl;
        return;
    }

    codecOption = NULL;
    av_dict_set(&codecOption, "profile", profile, 0);
    av_dict_set(&codecOption, "preset", "slow", 0);
    av_dict_set(&codecOption, "tune", "zerolatency", 0);

    ret = avcodec_open2(avccRtmpOutput, videoCodec, &codecOption);
    if (ret < 0) {
        std::cerr << "[FFMPEGOutput::Init] avformat_new_stream failed" << std::endl;
        return;
    }

    rtmpStream->codecpar->extradata = avccRtmpOutput->extradata;
    rtmpStream->codecpar->extradata_size = avccRtmpOutput->extradata_size;

    // 输出rtmp相关信息
    av_dump_format(avfcRtmpOutput, 0, rtmpOutputPath, 1);

    // 写视频头文件
    ret = avformat_write_header(avfcRtmpOutput, NULL);
    if (ret < 0) {
        std::cerr << "[FFMPEGOutput::init] avformat_write_header failed"
                << std::endl;
        return;
    }
    // 初始化一个AVStream结构体（用默认值填充）（此结构描述了解码（原始）音频或视频数据）
    videoFrame = av_frame_alloc();
    // 返回以给定参数存储图像所需的数据量的大小（以字节为单位）
    int frameBufSize = av_image_get_buffer_size(avccRtmpOutput->pix_fmt, avccRtmpOutput->width, avccRtmpOutput->height, 1);
    std::cout << "[FFMPEGOutput::Init] expected frame size " << frameBufSize
              << std::endl;

    videoFrame->width = avccRtmpOutput->width;
    videoFrame->height = avccRtmpOutput->height;
    videoFrame->format = avccRtmpOutput->pix_fmt;
    videoFrame->pts = 1;  //演示时间戳记，以time_base为单位（应向用户显示帧的时间）

    isValid = true;
}
// void FFMPEFOutput::SendFrame(const uint8_t* pdata){
//     PERF_TIMER();
//     // 根据指定的图像参数和提供的数组设置数据指针和行大小
//     int ret = av_image_fill_arrays(videoFrame->data, videoFrame->linesize, pdata,
//                          avccRtmpOutput->pix_fmt, avccRtmpOutput->width,
//                          avccRtmpOutput->height, 1);
//     // Supply a raw video or audio frame to the encoder
//     ret = avcodec_send_frame(avccRtmpOutput, videoFrame);
//     if (ret < 0) {
//         std::cerr << "[FFMPEGOutput::SendFrame] avcodec_send_frame failed"
//                 << std::endl;
//         return;
//     }
  
//     while (true) {
//         AVPacket pkt = {0};
//         av_init_packet(&pkt);
//         // Read encoded data from the encoder
//         ret = avcodec_receive_packet(avccRtmpOutput, &pkt);
  
//         if (ret < 0) {
//             av_packet_unref(&pkt);
//             return;
//         }
//         // 将数据包写入输出媒体文件，以确保正确的间隔
//         ret = av_interleaved_write_frame(avfcRtmpOutput, &pkt);
  
//         if (ret < 0) {
//             std::cerr << "[FFMPEGOutput::SendFrame] av_interleaved_write_frame failed"
//                     << std::endl;
//             return;
//         }
  
//         av_packet_unref(&pkt);
  
//         videoFrame->pts += av_rescale_q(1, avccRtmpOutput->time_base, rtmpStream->time_base);
//     }
// }

static void dontfree(void *opaque, uint8_t *data) {
  // tell ffmpeg dont free data
}
void FFMPEGOutput::SendRtmpFrame(void* pdata, int _size){
    ATLAS_LOG_INFO("start send rtmp frame");
    int ret = 0;
    AVPacket pkt = {0};
    av_init_packet(&pkt);
  
    pkt.pts = videoFrame->pts;
    pkt.dts = pkt.pts;
    pkt.flags = AV_PKT_FLAG_KEY;

    // 根据给出的pdata创建一个AVBuffer（对数据缓冲区的引用）
    pkt.buf = av_buffer_create(
          (uint8_t *)pdata, _size + AV_INPUT_BUFFER_PADDING_SIZE, dontfree, NULL, 0);
  
    pkt.data = (uint8_t *)pdata;
    pkt.size = _size;
  
    // 写视频帧
    ret = av_write_frame(avfcRtmpOutput, &pkt);
    if (ret < 0) {
        std::cerr << "[FFMPEGOutput::SendFrame] av_interleaved_write_frame failed"
                << std::endl;
        return;
    }
  
    // pkt.buf = nullptr;
  
    av_packet_unref(&pkt);
  
    videoFrame->pts += av_rescale_q(1, avccRtmpOutput->time_base, rtmpStream->time_base);
    ATLAS_LOG_INFO("[FFMPEGOutput::SendFrame] Sent Done");
}
void FFMPEGOutput::Destroy(){
    if(avfcRtmpOutput){
        av_write_trailer(avfcRtmpOutput);
        if(!(avfcRtmpOutput->flags & AVFMT_NOFILE)){
            avio_closep(&avfcRtmpOutput->pb);
        }
    }
    av_frame_free(&videoFrame);
    avcodec_close(avccRtmpOutput);
    avio_close(avfcRtmpOutput->pb);
    avformat_free_context(avfcRtmpOutput);
}
bool FFMPEGOutput::IsValid(){
    return isValid;
}