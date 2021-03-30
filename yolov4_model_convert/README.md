文件作用说明：

1.  dy_resize.py：onnx算子修改脚本 
2.  env.sh：ATC工具环境变量配置脚本
3.  parse_json.py： coco数据集标签json文件解析脚本 
4.  preprocess_yolov4_pytorch.py： 二进制数据集预处理脚本
5.  get_coco_info.py： yolov4.info生成脚本 
6.  bin_to_predict_yolov4_pytorch.py： benchmark输出bin文件解析脚本
7.  map_calculate.py： 精度统计脚本
8.  aipp.config JPEG输入配置文件
9.  require.txt：脚本运行所需的第三方库

推理端到端步骤：

（1） git clone 开源仓https://github.com/Tianxiaomo/pytorch-YOLOv4，并下载对应的权重文件， 修改**demo_pytorch2onnx.py**脚本生成onnx文件

```
git clone https://github.com/Tianxiaomo/pytorch-YOLOv4
python3 demo_pytorch2onnx.py yolov4.pth data/dog.jpg -1 80 608 608
```

（2）运行dy_resize.py修改生成的onnx文件

```
python3.7 dy_resize.py yolov4_-1_3_608_608_dynamic.onnx
```

（3）配置环境变量转换om模型

```
source env.sh
# 二进制输入
atc --model=yolov4_-1_3_608_608_dynamic_dbs.onnx --framework=5 --output=yolov4_bs1 --input_format=NCHW --log=info --soc_version=Ascend310 --input_shape="input:1,3,608,608" --out_nodes="Conv_495:0;Conv_518:0;Conv_541:0"
# JPEG输入
atc --model=yolov4_-1_3_608_608_dynamic_dbs.onnx --framework=5 --output=yolov4_bs1_with_aipp --input_format=NCHW --log=info --soc_version=Ascend310 --input_shape="input:1,3,608,608" --out_nodes="Conv_495:0;Conv_518:0;Conv_541:0" --insert_op_conf=aipp.config
```

（4）解析数据集

下载coco2014数据集val2014和label文件**instances_valminusminival2014.json**，运行**parse_json.py**解析数据集

```
python3.7 parse_json.py
```

生成coco2014.name和coco_2014.info以及gronud-truth文件夹

（5）数据预处理

运行脚本preprocess_yolov4_pytorch.py处理数据集

```
python3.7 preprocess_yolov4_pytorch.py coco_2014.info yolov4_bin
```

（6）benchmark推理

运行get_coco_info.py生成info文件

```
python3.7 get_coco_info.py yolo_coco_bin_tf coco_2014.info yolov4.info
```

执行benchmark命令，结果保存在同级目录 result/dumpOutput_device0/

```
# 二进制
./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -om_path=yolov4_bs1.om -input_width=608 -input_height=608 -input_text_path=yolov4.info -useDvpp=false -output_binary=true
# JPEG
./benchmark.x86_64 -model_type=vision -batch_size=1 -device_id=0 -om_path=yolov4_bs1_with_aipp.om -input_width=608 -input_height=608 -input_text_path=coco_2014.info -useDvpp=true -output_binary=true
```

（7）后处理

运行 bin_to_predict_yolov4_pytorch.py 解析模型输出

```
python3.7 bin_to_predict_yolov4_pytorch.py  --bin_data_path result/dumpOutput_device0/  --det_results_path  detection-results/ --origin_jpg_path /root/dataset/coco2014/val2014/ --coco_class_names /root/dataset/coco2014/coco2014.names
```

运行map_cauculate.py统计mAP值

```
python3 map_calculate.py --label_path  ./ground-truth  --npu_txt_path ./detection-results -na -np
```

