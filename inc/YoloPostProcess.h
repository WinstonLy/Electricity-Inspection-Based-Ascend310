#ifndef __YOLOV3POST_H__
#define __YOLOV3POST_H__

#include <algorithm>
#include <vector>
#include <memory>

const int CLASS_NUM = 80;
const int BIASES_NUM = 18; // Yolov3 anchors, generate from train data, coco dataset
const float BIASES[BIASES_NUM] = {12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401};
const float SCORE_THRESH = 0.4; // Threshold of confidence
const float OBJECTNESS_THRESH = 0.6; // Threshold of objectness value
const float IOU_THRESH = 0.6; // Non-Maximum Suppression threshold
const float COORDINATE_PARAM = 2.0;
const int YOLO_TYPE = 3;
const int ANCHOR_DIM = 3;
const int BOX_DIM = 4;

struct OutputLayer {
    int layerIdx;
    int width;
    int height;
    float anchors[6];
};

struct NetInfo {
    int anchorDim;
    int classNum;
    int bboxDim;
    int netWidth;
    int netHeight;
    std::vector<OutputLayer> outputLayers;
};

struct YoloImageInfo {
    int modelWidth;
    int modelHeight;
    int imgWidth;
    int imgHeight;
};

// Box information
struct DetectBox {
    float prob;
    int classID;
    float x;
    float y;
    float width;
    float height;
};

// Detect Info which could be transformed by DetectBox
struct ObjDetectInfo {
    float leftTopX;
    float leftTopY;
    float rightBotX;
    float rightBotY;
    float confidence;
    float classId;
};

// Realize the Yolo layer to get detiction object info
void Yolov3DetectionOutput(std::vector<std::shared_ptr<void>> featLayerData,
                           std::vector<ObjDetectInfo> &objInfos,
                           YoloImageInfo imgInfo);

#endif