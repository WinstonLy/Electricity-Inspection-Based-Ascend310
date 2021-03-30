/*
* @Author: winston
* @Date:   2021-03-15 21:22:17
* @Last Modified by:   winston
* @Last Modified time: 2021-03-29 22:19:09
* @Description: 
* @FilePath: /home/winston/AscendProjects/rtsp_dvpp_infer_dvpp_rtmp_test/atlas200dk_yolov4/atlas200dk_yolov4_test/src/YoloPostProcess.cpp 
*/

#include "YoloPostProcess.h"

#include <string>
#include <vector>
#include <iostream>
#include "FastMath.h"

namespace {
/*
 * @description: Initialize the Yolo layer
 * @param netInfo  Yolo layer info which contains anchors dim, bbox dim, class number, net width, net height and
                   3 outputlayer(13*13, 26*26, 52*52)
 * @param netWidth  model input width
 * @param netHeight  model input height
 */
void InitNetInfo(NetInfo& netInfo,
                 int netWidth,
                 int netHeight)
{
    netInfo.anchorDim = ANCHOR_DIM;
    netInfo.bboxDim = BOX_DIM;
    netInfo.classNum = CLASS_NUM;
    std:: cout << "anchorDim:" << netInfo.anchorDim << " bboxDim:" << netInfo.bboxDim 
               << " classNum" << netInfo.classNum << std::endl;
    netInfo.netWidth = 608;
    netInfo.netHeight = 608;
    const int featLayerNum = YOLO_TYPE;
    std::cout << "featLayerNum = " << featLayerNum << std::endl;
    const int minusOne = 1;
    const int biasesDim = 2;
    // yolov4 76*76 38*38 19*19,与yolov3的输出是相反的
    for (int i = 0; i < featLayerNum; ++i) {
        const int scale = 8 << i;    //yolov4
        // const int scale = 32 >> i; // yolov3
        OutputLayer outputLayer = {i, netWidth / scale, netHeight / scale, };
        int startIdx = (featLayerNum - minusOne - i) * netInfo.anchorDim * biasesDim;
        int endIdx = startIdx + netInfo.anchorDim * biasesDim;
        int idx = 0;
        for (int j = startIdx; j < endIdx; ++j) {
            outputLayer.anchors[idx++] = BIASES[j];
        }
        netInfo.outputLayers.push_back(outputLayer);
    }
    // std::cout << "init net info success" << std::endl;
}

/*
 * @description: Compute Intersection over Union value
 */
float BoxIou(DetectBox a, DetectBox b)
{
    float left = std::max(a.x - a.width / 2.f, b.x - b.width / 2.f);
    float right = std::min(a.x + a.width / 2.f, b.x + b.width / 2.f);
    float top = std::max(a.y - a.height / 2.f, b.y - b.height / 2.f);
    float bottom = std::min(a.y + a.height / 2.f, b.y + b.height / 2.f);
    if (top > bottom || left > right) { // If no intersection
        return 0.0f;
    }
    // intersection / union
    float area = (right - left) * (bottom - top);
    return area / (a.width * a.height + b.width * b.height - area);
}

/*
 * @description: Filter the Deteboxes, for each class, if two Deteboxes' IOU is greater than threshold,
                 erase the one with smaller confidence
 * @param dets  DetectBox vector where all DetectBoxes's confidences are greater than threshold
 * @param sortBoxes  DetectBox vector after filtering
 */
void FilterByIou(std::vector<DetectBox> dets, std::vector<DetectBox>& sortBoxes)
{
    for (unsigned int m = 0; m < dets.size(); ++m) {
        auto& item = dets[m];
        sortBoxes.push_back(item);
        for (unsigned int n = m + 1; n < dets.size(); ++n) {
            if (BoxIou(item, dets[n]) > IOU_THRESH) {
                dets.erase(dets.begin() + n);
                --n;
            }
        }
    }
}

/*
 * @description: Sort the DetectBox for each class and filter out the DetectBox with same object using IOU
 * @param detBoxes  DetectBox vector where all DetectBoxes's confidences are greater than threshold
 */
void NmsSort(std::vector<DetectBox>& detBoxes)
{
    std::vector<DetectBox> sortBoxes;
    std::vector<std::vector<DetectBox>> resClass;
    resClass.resize(CLASS_NUM);
    for (const auto& item: detBoxes) {
        resClass[item.classID].push_back(item);
    }
    for (int i = 0; i < CLASS_NUM; ++i) {
        auto& dets = resClass[i];
        if (dets.size() == 0) {
            continue;
        }
        std::sort(dets.begin(), dets.end(), [=](const DetectBox& a, const DetectBox& b) {
            return a.prob > b.prob;
        });
        FilterByIou(dets, sortBoxes);
    }
    detBoxes = std::move(sortBoxes);
}

/*
 * @description: Adjust the center point, box width and height of the prediction box based on the real image size
 * @param detBoxes  DetectBox vector where all DetectBoxes's confidences are greater than threshold
 * @param netWidth  Model input width
 * @param netHeight  Model input height
 * @param imWidth  Real image width
 * @param imHeight  Real image height
 */
void CorrectBbox(std::vector<DetectBox>& detBoxes, int netWidth, int netHeight, int imWidth, int imHeight)
{
    // correct box
    int newWidth;
    int newHeight;
    if ((static_cast<float>(netWidth) / imWidth) < (static_cast<float>(netHeight) / imHeight)) {
        newWidth = netWidth;
        newHeight = (imHeight * netWidth) / imWidth;
    } else {
        newHeight = netHeight;
        newWidth = (imWidth * netHeight) / imHeight;
    }
    for (auto& item : detBoxes) {
        item.x = (item.x * netWidth - (netWidth - newWidth) / 2.f) / newWidth;
        item.y = (item.y * netHeight - (netHeight - newHeight) / 2.f) / newHeight;
        item.width *= static_cast<float>(netWidth) / newWidth;
        item.height *= static_cast<float>(netHeight) / newHeight;
    }
    // std::cout << "correct box success" << std::endl;
}

/*
 * @description: Compare the confidences between 2 classes and get the larger one
 */
void CompareProb(int& classID, float& maxProb, float classProb, int classNum)
{
    if (classProb > maxProb) {
        maxProb = classProb;
        classID = classNum;
    }
}

/*
 * @description: Select the highest confidence class label for each predicted box and save into detBoxes with NHWC
 *               format
 * @param netout  The feature data which contains box coordinates, objectness value and confidence of each class
 * @param info  Yolo layer info which contains class number, box dim and so on
 * @param detBoxes  DetectBox vector where all DetectBoxes's confidences are greater than threshold
 * @param stride  Stride of output feature data
 * @param layer  Yolo output layer
 */
void SelectClassNHWC(std::shared_ptr<void> netout, NetInfo info, std::vector<DetectBox>& detBoxes, int stride,
    OutputLayer layer)
{
    const int offsetY = 1;
    const int offsetWidth = 2;
    const int offsetHeight = 3;
    const int biasesDim = 2;
    const int offsetBiases = 1;
    const int offsetObjectness = 1;
    fastmath::fastMath.Init();
    std::cout << "fast math init success" << std::endl;
    for (int j = 0; j < stride; j++) {
        for (int k = 0; k < info.anchorDim; k++) {
            int bIdx = (info.bboxDim + 1 + info.classNum) * stride * k + j; // begin index
            int oIdx = bIdx + info.bboxDim * stride; // objectness index
            // check obj
            float objectness = fastmath::Sigmoid(static_cast<float *>(netout.get())[oIdx]);
            
            // std::cout << "net out get " << std::endl;

            if (objectness <= OBJECTNESS_THRESH) {
                continue;
            }
            int classID = -1;
            float maxProb = SCORE_THRESH;
            float classProb;
            // Compare the confidence of the 3 anchors, select the largest one
            for (int c = 0; c < info.classNum; c++) {
                classProb = fastmath::Sigmoid(static_cast<float *>(netout.get())[bIdx +
                            (info.bboxDim + offsetObjectness + c)]) * stride * objectness;
                CompareProb(classID, maxProb, classProb, c);
            }
            if (classID >= 0) {
                DetectBox det = {};
                int row = j / layer.width;
                int col = j % layer.width;
                det.x = (col + fastmath::Sigmoid(static_cast<float *>(netout.get())[bIdx])) / layer.width;
                det.y = (row + fastmath::Sigmoid(static_cast<float *>(netout.get())[bIdx + stride])) / layer.height;
                det.width = fastmath::Exp(static_cast<float *>(netout.get())[bIdx + 2*stride]) *
                            layer.anchors[biasesDim * k]; // / info.netWidth;
                det.height = fastmath::Exp(static_cast<float *>(netout.get())[bIdx + 3*stride]) *
                             layer.anchors[biasesDim * k + offsetBiases]; // / info.netHeight;
                det.classID = classID;
                det.prob = maxProb;
                detBoxes.emplace_back(det);
            }
        }
    }


    // std::cout << "SelectClassNHWC success" << std::endl;
}

/*
 * @description: According to the yolo layer structure, encapsulate the anchor box data of each feature into detBoxes
 * @param featLayerData  Vector of 3 output feature data
 * @param info  Yolo layer info which contains anchors dim, bbox dim, class number, net width, net height and
                3 outputlayer(19*19, 38*38, 76*76)
 * @param detBoxes  DetectBox vector where all DetectBoxes's confidences are greater than threshold
 */
void GenerateBbox(std::vector<std::shared_ptr<void>> featLayerData, NetInfo info, std::vector<DetectBox>& detBoxes)
{
    // std::cut << "featLayerData size :"  << featLayerData.size() << std::endl;
    for (const auto& layer : info.outputLayers) {
        int stride = layer.width * layer.height; // 76*76 38*38 19*19
        std::cout << "stride : " << stride << std::endl;
        std::shared_ptr<void> netout = featLayerData[layer.layerIdx];
        SelectClassNHWC(netout, info, detBoxes, stride, layer);
    }

    std::cout << "generater box success" << std::endl;
}

/*
 * @description: Transform (x, y, w, h) data into (lx, ly, rx, ry), save into objInfos
 * @param detBoxes  DetectBox vector where all DetectBoxes's confidences are greater than threshold
 * @param objInfos  DetectBox vector after transformation
 * @param originWidth  Real image width
 * @param originHeight  Real image height
 */
void GetObjInfos(const std::vector<DetectBox>& detBoxes, std::vector<ObjDetectInfo>& objInfos, int originWidth,
    int originHeight)
{
    for (size_t k = 0; k < detBoxes.size(); k++) {
        if ((detBoxes[k].prob <= SCORE_THRESH) || (detBoxes[k].classID < 0)) {
            continue;
        }
        ObjDetectInfo objInfo = {};
        objInfo.classId = detBoxes[k].classID;
        objInfo.confidence = detBoxes[k].prob;
        objInfo.leftTopX = (detBoxes[k].x - detBoxes[k].width / COORDINATE_PARAM > 0) ?
                (float)((detBoxes[k].x - detBoxes[k].width / COORDINATE_PARAM) * originWidth) : 0;
        objInfo.leftTopY = (detBoxes[k].y - detBoxes[k].height / COORDINATE_PARAM > 0) ?
                (float)((detBoxes[k].y - detBoxes[k].height / COORDINATE_PARAM) * originHeight) : 0;
        objInfo.rightBotX = ((detBoxes[k].x + detBoxes[k].width / COORDINATE_PARAM) <= 1) ?
                (float)((detBoxes[k].x + detBoxes[k].width / COORDINATE_PARAM) * originWidth) : originWidth;
        objInfo.rightBotY = ((detBoxes[k].y + detBoxes[k].height / COORDINATE_PARAM) <= 1) ?
                (float)((detBoxes[k].y + detBoxes[k].height / COORDINATE_PARAM) * originHeight) : originHeight;
        objInfos.push_back(objInfo);
    }
}
}

/*
 * @description: Realize the Yolo layer to get detiction object info
 * @param featLayerData  Vector of 3 output feature data
 * @param objInfos  DetectBox vector after transformation
 * @param netWidth  Model input width
 * @param netHeight  Model input height
 * @param imgWidth  Real image width
 * @param imgHeight  Real image height
 */
void Yolov3DetectionOutput(std::vector<std::shared_ptr<void>> featLayerData,
                           std::vector<ObjDetectInfo>& objInfos,
                           YoloImageInfo imgInfo)
{
    static NetInfo netInfo;
    if (netInfo.outputLayers.empty()) {
        InitNetInfo(netInfo, 608, 608);
    }
    std::vector<DetectBox> detBoxes;
    GenerateBbox(featLayerData, netInfo, detBoxes);
    CorrectBbox(detBoxes, 608, 608, 2688, 1520);
    NmsSort(detBoxes);
    GetObjInfos(detBoxes, objInfos, 2688, 1520);
}
