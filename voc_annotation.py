# 这个文件是将VOC格式的标注文件生成一个yolo需要的标注文件，具体步骤如下：
# 可以参考https://blog.csdn.net/u012746060/article/details/81183006
#1.我们通过工具标注图片，生成的标注数据是一个voc格式的
#2.我们需要随机采样生成四（一般就三个）个文件文本文件（类似于D:\data\VOC2007\ImageSets\Main中的文件），
# 每个文件中表示训练/测试/验证等步骤对应的图片序号（这一步的代码不在本文件中）
#3.读取步骤2中的文件，然后依次读取对应图片的标注文件（voc中标注文件是xml格式）
# 然后得到三个真正用于yolo训练/测试/验证的标注文本文件，that all。

import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# 这是我自己的图片位置以及类别
my_classes=['batteryCar', 'person', 'head']
my_sets=[('20200311', 'train'), ('20200311', 'val'), ('20200311', 'test'),('20200311', 'trainval')]
my_wd='D:/data/myYOLOdata'

def convert_annotation(year, image_id, list_file):
    '''
    根据图片的ID写入该图片的gt数据
    :param year: VOC数据年份
    :param image_id: 图片id
    :param list_file: 要写入的文件
    :return: 无
    '''
    in_file = open('%s/VOC%s/Annotations/%s.xml'%(my_wd, year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in my_classes or int(difficult)==1:
            continue
        cls_id = my_classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

# wd = getcwd()
#
# for year, image_set in sets:
#     image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
#     list_file = open('%s_%s.txt'%(year, image_set), 'w')
#     for image_id in image_ids:
#         list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
#         convert_annotation(year, image_id, list_file)
#         list_file.write('\n')
#     list_file.close()


# 按照自己的文件生成label数据
for year, image_set in my_sets:
    image_ids = open('%s/VOC%s/ImageSets/Main/%s.txt'%(my_wd, year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')    #需要保存的YOLO形式的标注文件
    for image_id in image_ids:
        list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(my_wd, year, image_id))  #先写入图片的真实路径
        convert_annotation(year, image_id, list_file)   #再写入图片的GT信息
        list_file.write('\n')
    list_file.close()