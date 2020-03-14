# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```

For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

### Usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
---

4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.

## How to Train
下面介绍如何在自己的数据集上训练yolov3模型
参考[链接](https://blog.csdn.net/u012746060/article/details/81183006)
1.首先我们按照VOC格式准备准备数据集，例如我给自己的准备的数据集就是VOC20200311,其中包含了Annotations，ImageSets，JPEGImages，labels这四个传统的文件夹，并且也包含videoData（存放视频素材，图片素材就是截取其中的），mrconfig.xml（标注工具的配置文件），train.txt,test.txt（训练以及测试图片的相对路径）；

2.采用标注工具标定图片，我才用的是MRLabeler-master软件，当然也可以采用大家常用的labelImg，标注工具会生成标注label。在Annotations文件夹为每一张标注图片生成一个xml后缀的label文件（包含文件路径，各个目标种类等详细信息），在labels中为每一个文件生成txt后缀的label文件（仅仅包含种类以及gt位置），同时会在ImageSets/Main/文件夹下生成四个文件train.txt,test.txt,trainval.txt,val.txt，他们中写入的是样本的序号，表示训练，验证，测试等对应的图片序号；

3.修改并运行voc_annotation.py代码，生成本yolo代码可以使用的标注文件，分别为20200311_train.txt,20200311_test.txt,20200311_val.txt,20200311_trainval.txt,这些文件的每一行都表示一个样本图片的绝对路劲+gts的坐标和类别；

4.**存疑，我觉得这一步没有必要！！**修改yolo3.cfg文件，在文件中国搜索“yolo”，共出现三处，每次都要修改附近的filters（最后一层卷积层个数），classes（输出类别），random（修改为0表示关闭多尺度训练）；

5.修改voc_classes.txt（修改为自己的类别）和yolo_anchors（修改为自己聚类的anchors大小，也可以用原本的）

6.修改train.py中标注数据的路径以及epoch等代码，开始训练。

						
		

