#!/bin/bash
#scp -r ./data ./out ./model ./run_rtsp_vdec_save.sh ./runJpegDEInfer.sh HwHiAiUser@192.168.1.2:/home/HwHiAiUser/tmp/atlas200dk_yolov4_test/
#scp -r ./src/acl.json HwHiAiUser@192.168.1.2:/home/HwHiAiUser/tmp/atlas200dk_test/src/
scp -r ./out HwHiAiUser@192.168.1.2:/home/HwHiAiUser/atlas200_workspace/atlas200dk_yolov4_test/
