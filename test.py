#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image
import pandas as pd

from yolo3_MobileNet.utils import compose, letterbox_image, get_random_data
from yolo3_MobileNet.mobilenet import yolo_eval, yolo_body, yolo_loss, preprocess_true_boxes, tiny_yolo_body
from train_mobileNet import get_anchors, get_classes, create_model

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from functools import wraps

from keras.models import Model

from functools import reduce

from PIL import Image
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
gpu_num=1


class YOLO(object):
    def __init__(self):
        MY_PATH = '/Users/akbya/Downloads/'
        self.model_path = MY_PATH+'OrnithoMate/logs/4gen_stage_1.h5' # model path or trained weights path
        self.anchors_path = MY_PATH+ 'Ornithomate-main/yolo_anchors.txt'
        self.classes_path = MY_PATH+ 'OrnithoMate/classes.txt'
        self.score = 0.1
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (320, 320) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        '''to generate the bounding boxes'''
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting

        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        # hsv_tuples = [(x / len(self.class_names), 1., 1.)
        #               for x in range(len(self.class_names))]
        # self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # self.colors = list(
        #     map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        #         self.colors))
        # np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        # np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        # np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        # default arg
        # self.yolo_model->'model_data/yolo.h5'
        # self.anchors->'model_data/yolo_anchors.txt'-> 9 scales for anchors
        return boxes, scores, classes

    def detect_image(self, image):
        # start = timer()
        rects = []
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # tf.Session.run(fetches, feed_dict=None)
        # Runs the operations and evaluates the tensors in fetches.
        #
        # Args:
        # fetches: A single graph element, or a list of graph elements(described above).
        #
        # feed_dict: A dictionary that maps graph elements to values(described above).
        #
        # Returns:Either a single value if fetches is a single graph element, or a
        # list of values if fetches is a list(described above).
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
        #             size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # thickness = (image.size[0] + image.size[1]) // 500
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # label = '{} {:.2f}'.format(predicted_class, score)
            # draw = ImageDraw.Draw(image)
            # label_size = draw.textsize(label, font)

            y1, x1, y2, x2 = box
            y1 = max(0, np.floor(y1 + 0.5).astype('float32'))
            x1 = max(0, np.floor(x1 + 0.5).astype('float32'))
            y2 = min(image.size[1], np.floor(y2 + 0.5).astype('float32'))
            x2 = min(image.size[0], np.floor(x2 + 0.5).astype('float32'))
            # print(label, (x1, y1), (x2, y2))
            bbox = dict([("classe", predicted_class),("score",score),("x1",str(x1)),("y1", str(y1)),("x2", str(x2)),("y2", str(y2))])
            rects.append(bbox)

        #     if y1 - label_size[1] >= 0:
        #         text_origin = np.array([x1, y1 - label_size[1]])
        #     else:
        #         text_origin = np.array([x1, y1 + 1])
        #
        #     # My kingdom for a good redistributable image drawing library.
        #     for i in range(thickness):
        #         draw.rectangle(
        #             [x1 + i, y1 + i, x2 - i, y2 - i],
        #             outline=self.colors[c])
        #     draw.rectangle(
        #         [tuple(text_origin), tuple(text_origin + label_size)],
        #         fill=self.colors[c])
        #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        #     del draw
        #
        # end = timer()
        # print(str(end - start))
        return rects

    def close_session(self):
        self.sess.close()


if __name__ == '__main__':
    yolo = YOLO()
    test_data = pd.read_csv('/Users/akbya/Downloads/Ornithomate/test.txt', sep=",", header=None)
    test_data.columns = ["path", "xmin", "ymin", "xmax", "ymax", "actual_label"]
    predicted = []
    scores = []
    for index, row in test_data.iterrows():
        image = Image.open(row['path'])
        rects = yolo.detect_image(image)
        print(row['path'])
        print(rects)
        if len(rects) > 0:
            max_score=rects[0]["score"]
            max_i=0
            for i, rect in enumerate(rects):
                if rect["score"] > max_score:
                    max_score = rect["score"]
                    max_i = i
            predicted.append(rects[max_i]["classe"])
            scores.append(rects[max_i]["score"])
        elif len(rects) == 0:
            predicted.append("17")
            scores.append(0)
    test_data["predicted_label"] = predicted
    test_data["confidence"] = scores
    np.savetxt(r'/Users/akbya/Downloads/Ornithomate/4gen_predicted.txt', test_data.values, fmt='%s')
    yolo.close_session()