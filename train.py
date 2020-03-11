"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    annotation_path = 'train.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416, 416)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting

    # 创建模型（并且可以加载预训练模型参数）
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2,
                             weights_path='model_data/yolo_weights.h5')  # make sure you know what you freeze

    # 记录训练的中间参数
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    # Reduce learning rate when a metric has stopped improving.lr每次乘以factor
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    # Stop training when a monitored quantity has stopped improving.
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1 #验证集占总样本个数
    with open(annotation_path) as f:
        lines = f.readlines()   #lines样本路径组成的list
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    #不太看得明白训练的样本和loss是怎么设置的，借用参考的代码一段话：
    #“把目标当成一个输入，构成多输入模型，把loss写成一个层，作为最后的输出，搭建模型的时候，
    #就只需要将模型的output定义为loss，而compile的时候，
    #直接将loss设置为y_pred（因为模型的输出就是loss，所以y_pred就是loss），
    #无视y_true，训练的时候，y_true随便扔一个符合形状的数组进去就行了。”
    #确实该模型是把label的值（y_true）当做一个input层输入网络，最后的输出层就是loss，但是后面的话就看不懂了

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    # 在网络权重部分冻结的情况下训练
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                                   num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # 不冻结网络权重的情况下，对整个网络训练
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 32  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                                   num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=100,
                            initial_epoch=50,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    '''
    创建训练模型
    :param input_shape: 训练图片大小416*416
    :param anchors: （coco中）聚类得到的anchor list长度为9
    :param num_classes: （coco）目标类别个数
    :param load_pretrained: 是否加载与训练模型
    :param freeze_body: 冻结哪些层，0：不冻结，1：冻结前185层（应该是所有层），2：冻结除最后3层输出层外所有层
    :param weights_path:预训练权重路径
    :return: 返回的模型为Model([model_body.input, *y_true], model_loss)，
    也就是把图片和label值（y_true）当做input输入，loss当做output输出
    '''
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    # y_true是一个tensor的list，该list如下
    # [Tensor("input_1:0", shape=(?, 13, 13, 3, 25), dtype=float32),
    # Tensor("input_2:0", shape=(?, 26, 26, 3, 25), dtype=float32),
    # Tensor("input_3:0", shape=(?, 52, 52, 3, 25), dtype=float32)]
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # 这个表达式什么意思，lambda我清楚，但是yolo_loss是一个函数，并且lambda()()是什么意思
    # 按照yolo_loss函数参数的定义，这个表达式的意思是把后面的 output_shape=(1,), name='yolo_loss',
    # arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})([*model_body.output, *y_true]
    # 都看作是yolo_loss函数的参数
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])

    # 什么意思？input=[model_body.input, *y_true]？ output=model_loss
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l], \
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''
    data generator for fit_generator
    样本生成器函数，为model.fit_generator（）生成样本生成器函数
    生成器函数的输出应该为：一个形如(inputs ， targets) 的tuple或者一个形如(inputs， targets，sample_weight) 的tuple。
    :param annotation_lines:图片路径名组成的list
    :param batch_size:这里是32
    :param input_shape:这里是416*416
    :param anchors: anchors组成的list
    :param num_classes:
    :return:返回一个tuple，yield类似于有记忆功能的return，没有加括号，返回多个值默认为tuple
    返回值为：（[image_data, *y_true], np.zeros(batch_size)）,其中[image_data, *y_true]表示输入数据，np.zeros(batch_size)表示label数据，
             因为，这里对应的model是被设置为两个输入(图像数据，真实label数据)以及一个输出（loss结果）
    '''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            # 读取指点图片的图像数据和box数据，这个label数据是已经resized到input_shape（416,416）大小的
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n # i在0-n之间循环，表示当前采集的样本序号
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''
    样本生成器函数
    :param annotation_lines:图片路径名组成的list
    :param batch_size:这里是32
    :param input_shape:这里是416*416
    :param anchors: anchors组成的list
    :param num_classes:
    :return:（[image_data, *y_true], np.zeros(batch_size)）,其中[image_data, *y_true]表示输入数据，np.zeros(batch_size)表示label数据，
             因为，这里对应的model是被设置为两个输入(图像数据，真实label数据)以及一个输出（loss结果）
    '''
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    _main()
