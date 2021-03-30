#ifndef __FFMPEG_IN_OUT_H__
#define __FFMPEG_IN_OUT_H__

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
}

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

#include <functional>
#include <iostream>
#include <string>
#include "utils.h"
#include <fstream>
#include <time.h>
class FFMPEGInput{
public:
        FFMPEGInput();
        ~FFMPEGInput();

        void Destroy();
        void InputInit(const string inputPath);
        void Run();

        int GetWidth();
        int GetHeight();
        void RegisterHandler(std::function<void(AVPacket *)> handler);
        int GetFrame();
        acldvppStreamFormat GetProfile();
private:
	    AVFormatContext* avfcRtspInput;
        AVCodecParameters *avcpRtspInput;
        AVBSFContext *bsfc;
        AVCodecContext *decoderContext;
        const AVBitStreamFilter *bsfFilter;

        bool needBsf;
        
        int videoIndex;

        std::function<void(AVPacket *)> packetHandler;
};


class FFMPEGOutput{
public:
    FFMPEGOutput();
    ~FFMPEGOutput();
    void Init(int _height, int _width, int frameRate, int picFormat, std::string _name);
    void SendFrame(const uint32_t* pdata);
    void SendRtmpFrame(void* pdata, int _size);
    void Destroy();
    bool IsValid();
private:
	AVCodec* videoCodec;
	AVStream* rtmpStream;
	AVFormatContext* avfcRtmpOutput;
	AVCodecContext* avccRtmpOutput;
    AVFrame* videoFrame;
    AVDictionary* codecOption;

    int height;
    int width;
    std::string streamName;
    bool isValid;
};
#endif // __FFMPEG_IN_OUT_H__