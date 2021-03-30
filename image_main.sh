# @Author: winston
# @Date:   2021-03-10 16:42:49
# @Last Modified by:   winston
# @Last Modified time: 2021-03-30 09:48:26
#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/Ascend/acllib/lib64/:$LD_LIBRARY_PATH
#./build/rtsp_input_rtmp_output rtsp://admin:Admin402@192.168.2.64:554/h264/ch1/sub/av_stream rtmp://192.168.1.2:1935/myapp/stream1
./out/image_main