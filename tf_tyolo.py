#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import picamera
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, InputLayer
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K

from yad2k.yad2k_yolo import yolo_eval, voc_label

#############################
### Initialize 
#############################
width = 416
height = 416
r_w = 13
r_h = 13
r_n = 5
classes = 20
#############################

#############################
### tiny_yolo_model
#############################
def tiny_yolo_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(width, height, 3)))
    model.add(Conv2D(16, use_bias=False, data_format="channels_last",
                     padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(32, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(64, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(128, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(256, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2)))

    model.add(Conv2D(512, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(1, 1)))

    model.add(Conv2D(1024, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(1024, use_bias=False, data_format="channels_last", padding='same', kernel_size=(3, 3), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(125, use_bias=True, data_format="channels_last", padding='same', kernel_size=(1, 1), strides=(1, 1)))

    return model
#############################


#############################
### Main()
#############################
### Make Tiny-YOLO model
tiny_yolo_model = tiny_yolo_model()
tiny_yolo_model.load_weights('weight/tyolo.h5')
# tiny_yolo_model.summary()
# Camera
cap = picamera.PiCamera()
cap.resolution = (640,480)
#cap.framerate = 24

while True:
    ### input data
    #time.sleep(0.01)
    img = np.empty((480, 640, 3), dtype=np.uint8)
    cap.capture(img, 'bgr')
    # cv2.imshow('Input Image', img)
    resize_img = cv2.resize(img, dsize=(width,height), interpolation=cv2.INTER_CUBIC)
    image_data = np.array(resize_img, dtype='float32') / 255.0
    x = np.expand_dims(image_data, axis=0)
    ### predict
    # start_time = time.time()
    preds = tiny_yolo_model.predict(x)
    # exec_time = time.time() - start_time
    # print("time:{0}".format(exec_time)+"[sec]")
    # probs = np.zeros((r_h * r_w * r_n, classes+1), dtype=np.float)
    size = (640, 480)
    out_boxes, out_scores, out_classes = yolo_eval(preds, size, score_threshold = 0.3, iou_threshold = 0.5, classes = classes)
    
    for i in range(len(out_classes)):
        cls = out_classes[i]
        score = out_scores[i]
        box = out_boxes[i]
    
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(size[0], np.floor(right + 0.5).astype('int32'))
        print(voc_label[cls], score, (left, top), (right, bottom))
        lt = (left, top)
        rb = (right, bottom)
        red = (0, 0, 255) # B,G,R
        img = cv2.rectangle(img, lt, rb, red, 3)
        img = cv2.putText(img, str(voc_label[cls]), (left+5, top+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red) 
    
    cv2.imshow('Detect Image', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

print("finish")

#############################

#if __name__ == '__main__':
#    main()

