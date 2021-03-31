# Electricity-Inspection-Based-Ascend310

​		借助于Ascend310 AI处理器完成深度学习算法部署任务，应用背景为变电站电力巡检，基于YOLO v4算法模型对常见电力巡检目标进行检测，并充分利用Ascend310提供的DVPP等硬件支持能力来完成流媒体的传输、处理等任务，并对系统性能做出一定的优化。

## 1 网络模型的部署

​        目前只在Atlas200DK上完成开源YOLO v4网络的部署，由于YOLO含有mish算子，该算子在Atlas200DK现有的版本支持度高，仅在ONNX框架下支持，因此需要将Pytorch下的YOLO v4转换成ONNX，再转换为OM文件。

> **[YOLO v4论文地址]**：(https://arxiv.org/abs/2004.10934)

- 模型文件获取地址：（https://github.com/Tianxiaomo/pytorch-YOLOv4）

    >**输入输出数据**
    >
    >- 输入数据
    >
    >| 输入数据 | 分辨率  | 数据类型 | 数据排布格式 |
    >| :------- | ------- | -------- | ------------ |
    >| input    | 608*608 | RGB_FP32 | NCHW         |
    >
    >- 输出数据
    >
    >    输出数据分别对应模型的三个输出，shape为1x255x76x76，其中1为batch数，255则为3*85，表示每个cell预测3个bbox，85为4个坐标+1个置信度+80分类概率，后两个为特征图大小，其他两个特征图对应为38x38，19x19
    >
    >| 输出数据      | 大小（feature map） | 数据类型 | 数据排布格式 |
    >| ------------- | ------------------- | -------- | ------------ |
    >| feature_map_1 | -1x255x76x76        | FLOAT32  | NCHW         |
    >| feature_map_2 | -1x255x38x38        | FLOAT32  | NCHW         |
    >| feature_map_3 | -1x255x19x19        | FLOAT32  | NCHW         |
    >
    >

- 获取权重文件之后，修改修改demo_pytorch2onnx.py源码，只保留模型Backbone，去除不支持的后处理算子

    ```python
    def transform_to_onnx(weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W):
        model = Yolov4(n_classes=n_classes, inference=False)  # inference改为False即可去除后处理算子
        pretrained_dict = torch.load(weight_file, map_location=torch.device('cuda'))
        model.load_state_dict(pretrained_dict)
        input_names = ["input"]
        output_names = ['feature_map_1', 'feature_map_2', 'feature_map_3']  # 输出节点改为三个
    
        dynamic = False
        if batch_size <= 0:
            dynamic = True
    
        if dynamic:
            x = torch.randn((1, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
            onnx_file_name = "yolov4_-1_3_{}_{}_dynamic.onnx".format(IN_IMAGE_H, IN_IMAGE_W)
            dynamic_axes = {"input": {0: "-1"}, "feature_map_1": {0: "-1"},
                            "feature_map_2": {0: "-1"}, "feature_map_3": {0: "-1"}}
            # Export the model
            print('Export the onnx model ...')
            torch.onnx.export(model,
                              x,
                              onnx_file_name,
                              export_params=True,
                              opset_version=11,
                              do_constant_folding=True,
                              input_names=input_names, output_names=output_names,
                              dynamic_axes=dynamic_axes)
    
            print('Onnx model exporting done')
            return onnx_file_name
    ```

    运行脚本：

    ```python
    python3.7 demo_pytorch2onnx.py yolov4.pth data/dog.jpg -1 80 608 608
    ```

    >**转换失败可能原因**
    >
    >环境不匹配，缺少pytorch，安装pytorch步骤如下：
    >
    >- 在官网下载anaconda：https://www.anaconda.com/download/#linux
    >
    >- 进入anaconda安装包路径，输入命令进行安装
    >
    >    ```sh
    >    bash Anaconda3-2020.11-Linux-x86_64.sh
    >    ```
    >
    >- 安装结束之后测试
    >
    >    ```sh
    >    conda --version
    >    conda upgrade --all
    >    which python
    >    ```
    >
    >- 创建虚拟环境：`conda create -name ascend python=3.7.5`
    >
    >- 进入虚拟环境安装 prtorch  CPU版本
    >
    >    ```sh
    >    source activate ascend  deactivate退出
    >    根据具体的硬件环境对应得命令安装pytorch 
    >    ```
    >
    >    [pytorch官网](https://pytorch.org/)

- 替换resize算子，生成的yolov4_-1_3_608_608_dynamic_dbs.onnx可用ATC工具转换为离线om模型。

    ```sh
    # 替换Resize节点
    for i in range(len(model.graph.node)):
        n = model.graph.node[i]
        if n.op_type == "Resize":
            # print("Resize", i, n.input, n.output)
            model.graph.initializer.append(
                onnx.helper.make_tensor('scales{}'.format(i), onnx.TensorProto.FLOAT, [4], [1, 1, 2, 2])
            )
            newnode = onnx.helper.make_node(
                'Resize',
                name=n.name,
                inputs=ReplaceScales(n.input, 'scales{}'.format(i)),
                outputs=n.output,
                coordinate_transformation_mode='asymmetric',
                cubic_coeff_a=-0.75,
                mode='nearest',
                nearest_mode='floor'
            )
            model.graph.node.remove(model.graph.node[i])
            model.graph.node.insert(i, newnode)
            print("replace {} index {}".format(n.name, i))
    ```

    运行脚本：

    ```sh
    python3.7 dy_resize.py yolov4_-1_3_608_608_dynamic.onnx
    ```

- 离线模型转换

    - 配置环境变量：

        ```sh
        export install_path=/usr/local/Ascend/ascend-toolkit/latest
        export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
        export PYTHONPATH=${install_path}/atc/python/site-packages:$PYTHONPATH
        export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
        export ASCEND_OPP_PATH=${install_path}/opp
        ```

    - 模型转换

        out_node的设置根据模型而定，最后三个卷积层的输出节点名称

        ```sh
        atc --model=yolov4_-1_3_608_608_dynamic_dbs.onnx --framework=5 --output=yolov4 --input_format=NCHW --log=info --soc_version=Ascend310 --input_shape="input:1,3,608,608" --out_nodes="Conv_402:0;Conv_418:0;Conv_434:0" --insert_op_conf=aipp.config
        ```

        >aipp配置文件如下：
        >
        >```
        >aipp_op{
        >aipp_mode:static
        >input_format : YUV420SP_U8
        >
        >src_image_size_w : 608
        >src_image_size_h : 608
        >
        >crop: false
        >load_start_pos_h : 0
        >load_start_pos_w : 0
        >crop_size_w : 608
        >crop_size_h: 608
        >
        >csc_switch : true
        >rbuv_swap_switch : true
        >
        >
        >min_chn_0 : 0
        >min_chn_1 : 0
        >min_chn_2 : 0
        >var_reci_chn_0: 0.003921568627451
        >var_reci_chn_1: 0.003921568627451
        >var_reci_chn_2: 0.003921568627451
        >
        >
        >matrix_r0c0: 256
        >matrix_r0c1: 0
        >matrix_r0c2: 359
        >matrix_r1c0: 256
        >matrix_r1c1: -88
        >matrix_r1c2: -183
        >matrix_r2c0: 256
        >matrix_r2c1: 454
        >matrix_r2c2: 0
        >input_bias_0: 0
        >input_bias_1: 128
        >input_bias_2: 128
        >}
        >```

        

- 开发过程中出现的一些问题记录如下：

    1. [后处理代码开发问题](###后处理代码开发问题):face_with_thermometer:

        Ascend310目前版本并不支持YOLO v4的后处理代码，因此采用C++代码自己实现后处理代码，C++代码参考YOLO v3后处理C++代码和YOLO v4后处理Python代码进行编写测试。

    2. [出现opencv和ffmpeg动态链接库找不到:face_with_thermometer:](###openv和ffmpeg动态链接库找不到解决办法)

### 后处理代码开发问题

针对于YOLOv4的后处理代码，参考[华为Model zoo的ATC YOLO v4项目](https://ascend.huawei.com/zh/#/software/modelzoo/detail/1/abb7e641964c459398173248aa5353bc)进行测试。

该项目中提供了YOLO v4的后处理脚本（`bin_to_predict_yolov4_pytorch.py`)。由于模型转换的过程中去除了YOLO v4的后处理代码，因此在程序运行的过程中看不到相应的处理结果，为完成YOLO v4的后处理模块，采用以下步骤：

1. 首先采用的方案是利用yolo v3的后处理代码来进行替换，但是进行处理的结果发现不正确，初步定为解算不正确

2. 将模型推理的结果（在创建的模型输出的时候将三个`feature map`存放到一个vector数组中）保存到一个二进制文件中（`.bin`文件），将其拷贝到`ATC YOLO v4`的文件夹中利用python文件进行后处理，得到的输出结果正确，确定模型转换过程正确，推理结果正确，出错的环节是后处理（解算+NMS）。

3. 对比C++后处理代码和Python后处理代码，发现代码整体思路一致，唯一有问题的是进行解算的时候不知道输出的数据排布格式是什么样的。

4. 利用Pycharm工具调试来debug，发现排布格式为NCWH，如下图所示:shape为1x255x76x76，其中1为batch数，255则为3x85，表示每个cell预测3个bbox，85为4个坐标+1个置信度+80分类概率，后两个为特征图大小，其他两个特征图对应为38x38，19x19。以76x76的为例，特和曾图大小为76x76，通道数为255，依次分别是{x，y，w，h，iou，class1_iou, ......, class80_iou}。因此模型输出文件保存为二进制格式的存放方式为现存放x，再存放y，依次存放，最后存放80类的分类iou。这一点是在后处理解算过程中需要注意的点。

    <img src="./image/feature_map.jpg" style="zoom:55%;" />                                                                   <img src="./image/feature_map_c.jpg" style="zoom:45%;" />



### openv和ffmpeg动态链接库找不到解决办法

1. 首先确定是否将Alas200Dk上的相关目录拷贝到服务器上，主要有`/usr/lib/aarch64-linux-gnu`和`/home/HwHiAiUser/ascend_ddk/arm`，`/usr/lib64/`三个目录

2. 利用`sudo find / -name libopencv_...`类似的命令查找位置，将对应路径按照下面所示的步骤添加到环境变量中

3. 然后将这些动态链接库添加到相应的路径

    ```sh
    1 修改LD_LIBRARY_PATH，命令如下：
    
    	vi ~/.bashrc 
    
       在最后一行加入export LD_LIBRARY_PATH=/usr/lib64:/home/winston/Ascend/acllib/lib64:/home/winston/ascend_ddk/arm/lib:$LD_LIBRARY_PATH
    
    	source ~/.bashrc
    
       	sudo ldconfig
    
    2 修改/etc/ld.so.conf，命令如下：
    
       vim /etc/ld.so.conf.d/atlas.so.conf
    
       将动态链接库的路径添加在这个文件的最后。
       /usr/lib64
       /home/winston/Ascend/acllib/lib64
       /home/winston/ascend_ddk/arm/lib
    
       sudo ldconfig
       
    在开发环境编译的时候需要指定cmake编译器，涉及到交叉编译工具
    	cmake ../src -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ -DCMAKE_SKIP_RPATH=TRUE
    ```

- 使用ldconfig的出现以下问题：

    ```sh
    /sbin/ldconfig.real:/home/winston/ascend_ddk/lib/libprotobuf.so.19 is not a symbolic link
    ```

    解决办法如下：

    - 首先检查链接：`sudo ldconfig -v`

    - 然后找到错误的链接信息

        ```sh
        # /sbin/ldconfig.real
        /sbin/ldconfig.real:/home/winston/ascend_ddk/lib/libprotobuf.so.19 is not a symbolic link
        ```

    - 根据错误信息查看该文件软链接到哪个文件

        ```sh
        ls -lh /home/winston/ascend_ddk/arm/lib/libprotobuf.so.19
        lrwxrwxrwx 1 root root 21 Mar  5 15:45 /home/winston/ascend_ddk/arm/lib/libprotobuf.so.19 -> libprotobuf.so.19.0.0
        发现应该链接到一个文件
        ```

    - 建立软链接

        ```sh
        sudo ln -sf /home/winston/ascend_ddk/arm/lib/libprotobuf.so.19.0.0 /home/winston/ascend_ddk/arm/lib/libprotobuf.so.19
        ```

        