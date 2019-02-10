import logging
import math
import numpy as np
import os
import time

import cv2
import tensorflow as tf
from keras.models import load_model, model_from_json

from .EAST import locality_aware_nms as nms_locality
from .EAST import lanms as lanms
from .EAST.model import *
from .EAST.losses import *
from .EAST.data_processor import restore_rectangle

log = logging.getLogger(__name__)


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    log.debug('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer



class EASTDetector:

    def __init__(self
            , gpu_list='0'
            , model_path=os.path.join(os.path.dirname(__file__), 'EAST/model/EAST_IC15+13_model.h5')):

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

        if not os.path.isfile(model_path):
            log.info('loading pretrained model from Google Drive URL')
            os.mkdir(os.path.dirname(model_path))
            from scene_text.util import download_file_from_google_drive
            download_file_from_google_drive('1hfIzGuQn-xApDYiucMDZvOCosyAVwvku',
                model_path)
            download_file_from_google_drive('1gnkdCToYQfdU3ssaOareFTBr0Nz6u4rr',
                os.path.join(os.path.dirname(model_path), 'model.json'))

        log.info('loading model from %s' % model_path)
        json_file = open('/'.join(model_path.split('/')[0:-1]) + '/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json, custom_objects={'tf': tf, 'RESIZE_FACTOR': RESIZE_FACTOR})
        self.model.load_weights(model_path)

    def detect(self, image):
        start_time = time.time()
        img_resized, (ratio_h, ratio_w) = resize_image(image)
        img_resized = (img_resized / 127.5) - 1
        # feed image into model
        score_map, geo_map = self.model.predict(img_resized[np.newaxis, :, :, :])
        timer = {'net': 0, 'restore': 0, 'nms': 0}
        boxes, timer = detect(score_map=score_map, geo_map=geo_map, timer=timer)
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        return boxes


if __name__ == '__main__':
    main()
