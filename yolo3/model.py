"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    '''
    Darknet定义的卷积层，实际上调用了keras中的Conv2D()函数
    :param args: list/tuple参数，该参数直接输入Conv2D函数中,所以是（filter_nums, keral_size,stride,...）也就是Conv2D中的参数
    :param kwargs:dict参数，该参数也是直接输入Conv2D中
    :return:返回卷积层函数
    '''
    """Wrapper to set Darknet parameters for Convolution2D.
    定义了Conv2D()函数的**kwargs参数，
    """
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}  # 先定义一个dict
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'  # 再向该dict中添加key：value键对
    darknet_conv_kwargs.update(kwargs)  # 把kwargs更新到darknet_conv_kwargs
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    '''
        Darknet Convolution2D followed by BatchNormalization and LeakyReLU.
    :param args:
    :param kwargs:
    :return:
    '''

    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode

    # 在图片的上和左侧各添加一行0，然后再做3*3/2的卷积，完成下采样，
    # 依次来代替卷积函数自己的“same”模式
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # ((top_pad, bottom_pad), (left_pad, right_pad))
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)

    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    '''
    darknet-53主体部分，包括前面52个卷积层（不包括最后一层的averagePooling）
    正常情况下输入的是416*416图片，输出的是13*13*1024的特征图
    Darknent body having 52 Convolution2D layers
    :param x:
    :return:
    '''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    '''
    每一个输出层最后的卷积结构，主网络是3组1*1和3*3的卷积，再连接一个1*1的卷积层
    输出一个正常的目标检测结果y，还有一个引出去用作concat的特征层x
    :param x:
    :param num_filters:
    :param out_filters: 输出层的特征图channel个数
    :return:x,y 其中x是要引出去做concatenate然后输出的层，y是当前层的输出
    '''
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y  # x是要引出去做concatenate然后输出的层，y是当前层的输出


def yolo_body(inputs, num_anchors, num_classes):
    '''
    定义yolo的darknet-53模型
     :param inputs:输入层
    :param num_anchors:每一个输出层的anchors个数
    :param num_classes:类别个数
    :return:返回一个keras定义的函数式模型，一个输入层，三个输出层，那么这个model在keras中是怎么存储的呢
    经过分析model的每一层结构，发现，yolov3模型在keras中一共有252层，从input到output分别为0-251
    其中最后三层249,250,251分别表示最后三个conv2D卷积层，也就是输出层
    240-248这九层，表示分别连接三个输出层的DBL层（conv2D-BN-Relu）,但是这九层是三层conv2D到三层BN到三层Relu
    0-239则表示正常的从头到尾的yolo网络。
    224层表示输出为52*52大小的那层concat
    204层表示输出为26*26大小的那层concat
    0-184层表示去除FC层的darknet53的网络
    '''
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])  # layers[152]应该是26*26的那个特征图层的序号，
    # 因为每个卷积层是conv-》BN-》leakRelu三层串联，所以序号才这么大
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])


def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''
    搭建yolov3的tiny_body，其中只有两个尺寸的输出，也就是13*13和26*26
    然后该网络相比于darknet-53减少了重复的残差结构
    :param inputs:输入层
    :param num_anchors:每一个输出层的anchors个数
    :param num_classes:类别个数
    :return:返回一个keras定义的函数式模型
    '''
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)
    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)
    y1 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x2)
    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))([x2, x1])

    return Model(inputs, [y1, y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    '''
    根据某个输出层，提取出output检测框信息
    :param feats: 某个输出层特征图，（以下用13*13*255的输出层举例）
    :param anchors:该输出层对应的anchor list中的序号
    :param num_classes:类别个数
    :param input_shape:原始图片的大小
    :param calc_loss:是否计算loss
    :return: calc_loss关闭时输出正常的预测的box信息，
        if calc_loss == True:
        return grid, feats, box_xy, box_wh
        grid表示输出层grid每个cell的坐标
        feats：表示网络的输出层，是网络的原始输出，参考yolov2的loss，也就是输出的是预测框的（tx,ty,tw,th,confidence,classes）
        box_xy, box_wh:是处理之后的预测框坐标，相当于bx,by,bw,bh归一化之后的坐标
                    表示坐标是相对于整个原始图像左上角和wh的归一化后的坐标
        else
        return box_xy, box_wh, box_confidence, box_class_probs
    '''
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width,[13,13]

    # #此时的shape是[13,13,1,1],它的值大概是[[[[0]],[[0]],...,[[0]]],[[[1]],[[1]],...,[[1]]],...]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])

    # grid是数值为0~12的全遍历二元组，表示输出层grid的坐标，如果输入的是最后一层网络的结果，则grid的结构是(13, 13, 1, 2)
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    # 起始点xy：将feats中xy的值，经过sigmoid归一化，再加上相应的grid的二元组，再除以网格边长，归一化
    # 这样或得的xy就是相对于整个原始图像左上角起点的归一化之后的偏移量
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))

    # 宽高wh：将feats中wh的值，经过exp正值化(也就是e的指数函数，可能wh的输出是经过ln函数处理过的)，
    # 再乘以anchors_tensor的anchor box，再除以图片宽高，归一化，这里的xywh的输出用yolov2的loss。
    # 这样获得的就是wh相比于整个原始图像wh的归一化值
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''
    yolo实际上进行检测的图片大小是416*416的，但实际输入的并不是，
    因此输入的图片首先会被填充128的灰度，再resize到416*416大小，最后再送入检测。
    本函数则是将yolo获得的检测框映射到原始图片上
    Get corrected boxes
    '''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                                                                anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    '''
    根据YOLO网络的输出层，提取得到的目标检测框和置信度等
    :param yolo_outputs: yolo的输出层
    :param anchors: 9个标准的anchor大小
    :param num_classes:类别个数
    :param image_shape:
    :param max_boxes:检测框的最大个数
    :param score_threshold:
    :param iou_threshold:
    :return:boxes_, scores_, classes_每一个框的置信度和类别
    '''
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        # 提取每一层的box和box_scores
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''
    Preprocess true boxes to training input format
    把样本的GT数据转换成可以用于训练的数据？
    :param true_boxes: 归一化到416*416大小后gt数据，nparray, shape=(batch_size, T, 5)，T是每张图GT的最大个数，
            T为20，前面表示GT，后面用0填充
            Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    :param input_shape: 训练输入的图片大小（416,416），array-like, hw, multiples of 32
    :param anchors: anchor list，array, shape=(N, 2), wh
    :param num_classes: 类别个数integer
    :return: 用于训练的标记gt数据。y_true: list of array, shape like yolo_outputs, xywh are reletive value
    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # 输出层个数，default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2   #box的中心坐标
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  #box的wh

    # 把true_boxes的box坐标有原来的（xmin,ymin,xmax,ymax）变成了（xcenter,ycenter,w,h）
    # 并除以图片大小（416*416）进行归一化
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0] #batch_size

    # grid_shapes是<class 'list'>，这个与yolo_loss（）函数中的grid_shapes是一样的
    # 形状为[array([13, 13], dtype=int32), array([26, 26], dtype=int32), array([52, 52], dtype=int32)]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]

    # y_true应该就是要传入网络的label，是一个有三个数组的list，每个数组形状为
    # (batch_size,13,13,3,8)，(batch_size,26,26,3,8)，(batch_size,52,52,3,8)
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)    #把anchors数组从原来的的shape（9,2）变为（1,9,2）
    anchor_maxes = anchors / 2. # anchors的值除以2，shape=(1,9,2)
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0   #boxes_wh[..., 0]是每一个gt的w数据，valid_mask是一个bool数组，表示GT不为0的序号

    for b in range(m):#对序号为b的样本提取gt
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]] #wh就是gt不是0的标注框的宽和高，shape=（gt_nums，2）
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2) #shape=(gt_nums,1,2)
        box_maxes = wh / 2. #wh的值除以2，shape=（gt_nums,1,2）
        box_mins = -box_maxes
        # 这一段代码是假设anchor与gts中心点重合，然后计算每一个gt与每一个anchor的IOU，并找到与gt重合度最高的anchor
        intersect_mins = np.maximum(box_mins, anchor_mins)  #np.maximum()是比较两数组返回最大值，如果数组形状不一样，则进行广播操作(broadcast)，本代码是形状为（3,1,2）和（1,9,2）比较后变为（3,9,2），3是gts的个数
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)#同上
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)   #np.argmax()返回最大值对应的序号，axis=-1表示从最后一个维度判断

        # 下面的代码的意思大概是
        # 首先根据gt所匹配的IOU最大的anchor得到best_anchor
        # yolov3的每一个输出层匹配三个尺寸的anchor，根据best_anchor可以找到对应的输出层，也就是说每个gt只在某一个输出层有对应的预测结果
        # 接着根据gt的坐标计算其左上角坐标在对应输出层的位置，例如某个gt为[0.31,0.50,0.43,0.62,0]5个值分别为xmin,ymin,xmax,ymax,class
        # 则其best_anchor是序号为7的anchor，对应第一层输出也就是输出大小为13*13的那层，并根据gt的左上角坐标计算
        # 得i=4,j=6
        # k表示序号为7的anchor在当前输出层匹配的3个anchor（分别为序号为6,7,8）中的序号，因此k=1
        # c表示类别序号，所以本gt的c=0

        # 再来说一说y_true这个list是怎么赋值的
        # y_true是有3个数组组成，其形状为[（32,13,13,3,8）,(32,26,26,3,8),(32,52,52,3,8)]分别表示（batch_size，output_size,output_size,anchor_num,4个坐标+1个置信度+3个类别）
        # 还是以本gt举例。赋值的过程为，在y_true[l][b, j, i, k,...]的位置写上对应gt信息，注意这里j表示y坐标，i表示x坐标，
        # 这里的ij对应的xy就是gt的中心，这与YOLOv3论文中说的是“目标的中心位于哪个grid，则就这个grid预测该目标”相符合
        for t, n in enumerate(best_anchor): #t是当前样本图片（序号为b）的gt序号，n是其对应的best_anchor序号
            for l in range(num_layers): # l是当前anchor应该由第l层输出层负责。第一层到第三输出层为分别为13*13，26*26,52*52
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')   # gt的xmin在对应输出层的坐标
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')   # gt的ymin在对应输出层的坐标
                    k = anchor_mask[l].index(n) #best_anchor在当前输出层中三个anchors中的序号
                    c = true_boxes[b, t, 4].astype('int32') #gt的类别
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]  #写上gt位置信息
                    y_true[l][b, j, i, k, 4] = 1    # 写上gt的置信度信息，也就是有目标为1，无目标为0
                    y_true[l][b, j, i, k, 5 + c] = 1    #写上类别信息，也就是对应类别为1，其他类别为0

    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''
    计算yolo loss tensor
    :param args: 就是这个参数[*model_body.output, *y_true]，而且model_body.output是一个长度为3的list，
                *model_body.output表示该list的内容,这是一个len为3的list，y_true也是，只不过它表示label
    :param anchors:聚类得到的anchor list
    :param num_classes:类别个数
    :param ignore_thresh:过滤置信度的阈值
    :param print_loss:
    :return:返回loss tensor
    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = args[:num_layers]  # 三个输出依次为[（None，13,13,255）,（None，26,26,255）,（None，52,52,255）]
    y_true = args[num_layers:]  # y_true是被当做一个输入层输入的，就是我们在处理gt生成可训练的label时生成的，同yolo_outputs，也是一个长度为3的list
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))  # 原始图片的大小

    # grid_shapes=[(13,13),(26,26),(52,52)]?
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]  # 这个mask就是三个输出层对应grid是否有对应的gt，也就是目标的置信度label
        true_class_probs = y_true[l][..., 5:]   #每个grid对应的类别

        # grid是第l层输出层cell的坐标，raw_pre是该输出层的原始数据，pred_xy, pred_wh则是raw_pre处理之后表示预测box坐标数据
        # 网络原本的输出的结果参考YOLOv2的loss，网络原始输出的是tx,ty,tw,th,confidence，class
        # 然后经过yolo_head（）处理之后，在训练的时候，对网络的输出做了如下的变化变为：
        # 对于xy的值：经过sigmoid归一化，再加上相应的grid的二元组，得到bx,by,再除以网格边长，归一化
        # 对于宽高wh：将feats中wh的值，经过exp正值化(也就是e的指数函数)，再乘以anchors_tensor的anchor box，得到bw，bh，再除以图片宽高，归一化，
        #            wy和wh最后一步归一化的目的是把数据集中到（0,1）区间方便训练，在归一化之前得到的bx,by,bw.bh就是预测框真正的坐标
        # 也就是说，网络直接输出的预测框结果是tx,ty,tw,th，经过yolo_head()处理后，得到bx,by,bw.bh归一化之后的预测框结果pred_xy, pred_wh，
        # 我们的label是y_true中gt的位置是用输入图片大小416归一化后的（xmin,ymin,xmax,ymax）
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])  # K.concatenate默认是将最后一个维度并联在一起

        #  y_true[l][..., :2] * grid_shapes[l][::-1]的结果是grid中每一个cell对应gt的中心点坐标
        #  所以raw_true_xy是sigmoid(tx)
        # 这就有意思了，网络的预测值raw_pred中xy的值是tx，ty，而对应的label是sigmoid(tx)
        # 我理解的是xy的loss采用K.binary_crossentropy（）也就是二元交叉熵来计算loss,这个公式会对预测值先做一个sigmoid处理
        # 是的了！在预测confidence和class部分也用的K.binary_crossentropy（）函数计算loss
        # 这就说明confidence和class的预测值也需要经过sigmoid之后才是真正的置信度和类别
        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid

        # y_true[l][..., 2:4]* input_shape[::-1]就是cell对应gt的wh，然后除以anchors[anchor_mask[l]]
        # 也就是除以当前grid对应的三个anchor的wh，这样得到的就是bh/ph,然后取对数，得到的
        # raw_true_wh是th和tw，之后的k.switch是防止log的值不存在，用0代替
        # 这样网络的预测值raw_pred中wh的值是tw,th同时对应的label也是th和tw
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf

        # 乘机项是grid中保存的gt的w*h，表示面积越大其权重越低
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        # 注意计算loss用的predict数据还是raw_pred数据，也就是网络最原始的输出，而不是处理之后的pred_xy, pred_wh
        # label则是y_true处理之后
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                            message='loss: ')
    return loss
