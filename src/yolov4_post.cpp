/*
* @Author: winston
* @Date:   2021-03-15 21:22:17
* @Last Modified by:   WinstonLy
* @Last Modified time: 2021-04-10 17:04:26
* @Description: 
* @FilePath: /home/winston/AscendProjects/rtsp_dvpp_infer_dvpp_rtmp_test/atlas200dk_yolov4/Electricity-Inspection-Based-Ascend310/src/yolov4_post.cpp 
*/

#include "yolov4_post.h"

#include <string>
#include <vector>
#include <iostream>
#include "fast_math.h"

namespace {
const static std::vector<std::string> yolov4Label = {
"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", 
"truck", "boat", "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", 
"bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", 
"zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
"skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove", 
"skateboard", "surfboard", "tennis_racket", "bottle", "wine_glass", "cup", "fork", 
"knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", 
"carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch", "potted_plant", 
"bed", "dining_table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
"cell_phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", 
"clock", "vase", "scissors", "teddy_bear", "hair_drier", "toothbrush  "};
}

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
    // std:: cout << "anchorDim:" << netInfo.anchorDim << " bboxDim:" << netInfo.bboxDim 
    //            << " classNum" << netInfo.classNum << std::endl;
    netInfo.netWidth = netWidth;
    netInfo.netHeight = netHeight;
    const int featLayerNum = YOLO_TYPE;
    // std::cout << "featLayerNum = " << featLayerNum << std::endl;
    const int minusOne = 1;
    const int biasesDim = 2;
    // yolov4 76*76 38*38 19*19,与yolov3的输出是相反的
    for (int i = 0; i < featLayerNum; ++i) {
        const int scale = 8 << i;    //yolov4
        // const int scale = 32 >> i; // yolov3
        // std::cout << "===============feature map stride: " << netWidth / scale << std::endl;
        // std::cout << "===============anchors :";
        
        // 模型推理输出排布格式：x,y,w,h,iou,class_iou1...class_iou80；也就是先存x，存完之后存y，依次类推
        int stride = netWidth / scale;
        OutputLayer outputLayer = {i, netWidth / scale, netHeight / scale, };
        int startIdx = i * netInfo.anchorDim * biasesDim;
        int endIdx = startIdx + netInfo.anchorDim * biasesDim;
        int idx = 0;
        for (int j = startIdx; j < endIdx; ++j) {
            outputLayer.anchors[idx++] = BIASES[j];
            
        }
        // std::cout << std::endl;
        // for(int m = 0; m < 6; ++m){
        //     std::cout << outputLayer.anchors[m] << " ";
        // }
        std::cout << std::endl;
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
    // std::cout << "fast math init success" << std::endl;
    int flag = 1;
    for (int j = 0; j < stride; j++) {
        for (int k = 0; k < info.anchorDim; k++) {
            int bIdx = k * stride * (info.bboxDim + 1 + info.classNum)
                + j;  // begin index
            int oIdx = bIdx + info.bboxDim*stride; // objectness index
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
                            (info.bboxDim + offsetObjectness + c)* stride]);
                CompareProb(classID, maxProb, classProb, c);
            }

            if (classID >= 0) {
                DetectBox det = {};
                int row = j / layer.width;
                int col = j % layer.width;
                det.x = (col + fastmath::Sigmoid(static_cast<float *>(netout.get())[bIdx])) / layer.width;
                det.y = (row + fastmath::Sigmoid(static_cast<float *>(netout.get())[bIdx + offsetY * stride])) / layer.height;
                det.width = fastmath::Exp(static_cast<float *>(netout.get())[bIdx + offsetWidth * stride]) *
                            layer.anchors[biasesDim * k] / info.netWidth;
                det.height = fastmath::Exp(static_cast<float *>(netout.get())[bIdx + offsetHeight * stride]) *
                             layer.anchors[biasesDim * k + offsetBiases]/ info.netHeight;
                det.classID = classID;
                det.prob = maxProb * objectness;
                detBoxes.emplace_back(det);
            }
            // if(stride == 361 && flag <= 50){

            //     std::cout << "x: " << static_cast<float *>(netout.get())[bIdx] << "y: " << static_cast<float *>(netout.get())[bIdx + offsetY] << std::endl;
            //     // flag = 0;
            //     flag++;
            // }
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
        // std::cout << "stride : " << stride << std::endl;
        std::shared_ptr<void> netout = featLayerData[layer.layerIdx];
        SelectClassNHWC(netout, info, detBoxes, stride, layer);
    }

    // std::cout << "generater box success" << std::endl;
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


/*
 * @description: Realize the Yolo layer to get detiction object info
 * @param featLayerData  Vector of 3 output feature data
 * @param objInfos  DetectBox vector after transformation
 * @param netWidth  Model input width
 * @param netHeight  Model input height
 * @param imgWidth  Real image width
 * @param imgHeight  Real image height
 */
void Yolov4DetectionOutput(std::vector<std::shared_ptr<void>> featLayerData,
                           std::vector<ObjDetectInfo>& objInfos,
                           YoloImageInfo imgInfo)
{
    static NetInfo netInfo;
    if (netInfo.outputLayers.empty()) {
        InitNetInfo(netInfo, imgInfo.modelWidth, imgInfo.modelHeight);
    }
    std::vector<DetectBox> detBoxes;

    GenerateBbox(featLayerData, netInfo, detBoxes);
    
    CorrectBbox(detBoxes, imgInfo.modelWidth, imgInfo.modelHeight, imgInfo.imgWidth, imgInfo.imgHeight);
    
    NmsSort(detBoxes);

    GetObjInfos(detBoxes, objInfos, imgInfo.imgWidth, imgInfo.imgHeight);
}





aclError WriteResult(const std::vector<ObjDetectInfo> &objInfos, int frameIndex)
{
    std::string timeString;
    GetCurTimeString(timeString);
    // Create result file under result directory
    // SetFileDefaultUmask();
    
    std::string resultName = "./out/result/result_" + std::to_string(frameIndex);
    std::string fileNameSave = "./out/result/result_" + std::to_string(frameIndex);

    std::ofstream tfile(resultName, std::ios::app);
    // Check result file validity
    if (tfile.fail()) {
        ATLAS_LOG_ERROR("Failed to open result file: %s", resultName.c_str());
        return ACL_ERROR_MEMORY_ADDRESS_UNALIGNED;
    }
    tfile.seekp(0, tfile.end);
    size_t dstFileSize = tfile.tellp();
    tfile.close();
    // if (dstFileSize > FILE_SIZE) {
    //     if (access(resultBakName_.c_str(), 0) == APP_ERR_OK) {
    //         APP_ERROR ret = remove(resultBakName_.c_str());
    //         if (ret != APP_ERR_OK) {
    //             LogError << "remove " << resultBakName_ << " failed." << std::endl;
    //             return ret;
    //         }
    //     }
    //     APP_ERROR ret = rename(resultName.c_str(), resultBakName_.c_str());
    //     if (ret != APP_ERR_OK) {
    //         LogError << "rename " << resultName << " failed." << std::endl;
    //         return ret;
    //     }
    // }
    tfile.open(resultName, std::ios::out);
    if (tfile.fail()) {
       ATLAS_LOG_ERROR("Failed to open result file: %s", resultName.c_str());
        return ACL_ERROR_MEMORY_ADDRESS_UNALIGNED;
    }
    tfile << "[Date:" << timeString << " Frame:" << frameIndex
          << "] Object detected number is " << objInfos.size() << std::endl;
    // Write inference result into file
    std::cout <<"[Date:" << timeString <<" Frame:" << frameIndex
          << "] Object detected number is " << objInfos.size() << std::endl;
    
    for (uint32_t i = 0; i < objInfos.size(); i++) {
        tfile << "#Obj" << i << ", " << "box(" << objInfos[i].leftTopX << ", " << objInfos[i].leftTopY << ", "
              << objInfos[i].rightBotX << ", " << objInfos[i].rightBotY << ") "
              << " confidence: " << objInfos[i].confidence << "  lable: " << objInfos[i].classId << std::endl;
    }

    // for (uint32_t i = 0; i < objInfos.size(); i++) {
    //     std::cout << "#Obj" << i << ", " << "box(" << objInfos[i].leftTopX << ", " << objInfos[i].leftTopY << ", "
    //           << objInfos[i].rightBotX << ", " << objInfos[i].rightBotY << ") "
    //           << " confidence: " << objInfos[i].confidence << "  lable: " << objInfos[i].classId << std::endl;
    // }
                           
    tfile << std::endl;
    tfile.close();
    return ACL_ERROR_NONE;
}


aclError GetObjectInfoYolo(std::vector<RawData> &modelOutput, std::vector<ObjDetectInfo> &objInfos,
                           int frameWidth, int frameHeight, std::vector<void*>& outputDataBuffers,
                           int modelWidth, int modelHeight)
{
    std::vector<std::shared_ptr<void>> hostPtr;
    
    for (size_t i = 0; i < modelOutput.size(); i++) {
        void *hostPtrBuffer = outputDataBuffers[i];
        std::shared_ptr<void> hostPtrBufferManager(hostPtrBuffer, [](void *) {});
        aclError ret = aclrtMemcpy(hostPtrBuffer, modelOutput[i].lenOfByte, modelOutput[i].data,
            modelOutput[i].lenOfByte, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret!= ACL_ERROR_NONE || hostPtrBuffer == nullptr) {
            ATLAS_LOG_ERROR("Failed to copy output buffer of model from device to host, ret =%d", ret);
            return ret;
        }
        hostPtr.push_back(hostPtrBufferManager);
    }

    YoloImageInfo yoloImageInfo  = {};
    yoloImageInfo.imgWidth       = frameWidth;
    yoloImageInfo.imgHeight      = frameHeight;
    yoloImageInfo.modelWidth     = modelWidth;
    yoloImageInfo.modelHeight    = modelHeight;
    Yolov4DetectionOutput(hostPtr, objInfos, yoloImageInfo);
    return ACL_ERROR_NONE;
}





vector<DetectionResult> PostProcessYolov4(vector<size_t>& inferOutSizes, vector<void*>& inferOutBuffer,
                                          int frameWidth, int frameHeight, int modelWidth, int modelHeight)
{
    std::vector<RawData> Output;
    // std::vector<void*> inferOutputBuffers = GetOutputDataBuffers();
    // std::vector<size_t> inferOutputSizes = GetOutputBufferSizes();
    int size = inferOutSizes.size();
    for(int i = 0; i < size; ++i){
        RawData rawData = RawData();
        rawData.data= inferOutBuffer[i];
        rawData.lenOfByte = inferOutSizes[i];
        Output.push_back(std::move(rawData));
    }

    size_t outputLength = Output.size();
    if(outputLength <= 0){
        ATLAS_LOG_ERROR("Failed to get model infer output data");
        // return ACL_ERROR_MEMORY_ADDRESS_UNALIGNED;
    }

    std::vector<ObjDetectInfo> objInfos;
    static int frameIndex = 0;
    aclError ret = GetObjectInfoYolo(Output, objInfos, frameWidth, frameHeight, inferOutBuffer, modelWidth, modelHeight);
    if(ret != ACL_ERROR_NONE){
        ATLAS_LOG_ERROR("Falied to get TensorFlow model output, ret = %d", ret);
        // return ret;
    }
    vector<DetectionResult> detectResults;
    for(int i = 0; i < objInfos.size(); i++){
        DetectionResult oneResult;
        Point point_lt, point_rb;
        //get the confidence of the detected object. Anything less than 0.8 is considered invalid
        uint32_t score = (uint32_t)(objInfos[i].confidence * 100);
        if (score < 80) continue;
        ATLAS_LOG_INFO("infer score %d", score);
        //get the frame coordinates and converts them to the coordinates on the original frame
        oneResult.lt.x = objInfos[i].leftTopX;
        oneResult.lt.y = objInfos[i].leftTopY;
        oneResult.rb.x = objInfos[i].rightBotX;
        oneResult.rb.y = objInfos[i].rightBotY;
        //Construct a string that marks the object: object name + confidence
        uint32_t objIndex = (uint32_t)objInfos[i].classId;
        oneResult.result_text = yolov4Label[objIndex] + std::to_string(score) + "\%";
        ATLAS_LOG_INFO("%d %d %d %d %s\n", oneResult.lt.x, oneResult.lt.y,
         oneResult.rb.x, oneResult.rb.y, oneResult.result_text.c_str());
        

        detectResults.emplace_back(oneResult);
    }
 
    

    // // 手动提取3个feature，用python文件解析
    // static int imageNum = 0;
    // int tensorNum = 1;
    // for(int i = 0; i < outputDataBuffers.size(); ++i){
    //     std::string fileNameSave = "./results/output" + std::to_string(imageNum) + "_" 
    //                                 + std::to_string(tensorNum ) + ".bin";
    //     FILE* output = fopen(fileNameSave.c_str(), "wb+");
        
    //     void* data = outputDataBuffers[i];
    //     int sizeData = outputDataBufferSizes[i];
    //     size_t num = fwrite(data, 1, sizeData, output);
    //     std::cout << "write sizeData 0 : " << num << std::endl;
        
        
        
    //     fclose(output);

    //     if(tensorNum == 3){
    //         break;
    //     }
    //     ++tensorNum;

    // }
    // ++imageNum;
   

    return detectResults;
    // ret = WriteResult(objInfos, frameIndex);
    // if(ret != ACL_ERROR_NONE){
    //     ATLAS_LOG_ERROR("Failed to wrtie result, ret = %d", ret);
    //     return ret;
    // }
    // ++frameIndex;

    // return ACL_ERROR_NONE;

}


