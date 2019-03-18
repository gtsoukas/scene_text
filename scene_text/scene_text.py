import argparse
import glob
import logging
from math import atan2, degrees, fabs, sin, radians, cos
import numpy as np
import os

import cv2

from scene_text.detector import EASTDetector
from scene_text.recognizer import MORANRecognizer

log = logging.getLogger(__name__)

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)
    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgOut = imgRotation[int(pt1[1]):int(pt3[1]), int(pt1[0]):int(pt3[0])]
    height, width = imgOut.shape[:2]
    return imgOut


class AllWordsRecognizer:
    """Pipeline for detection and recognition of all words in an image"""

    def __init__(self, text_direction='ltr'):
        self.text_direction = text_direction
        self.detector = EASTDetector()
        self.recognizer = MORANRecognizer()

    def get_all_words(self, image):
        """Return lists of words and corresponding boxes"""

        log.debug("start processing image of shape {}".format(image.shape))

        boxes = self.detector.detect(image)

        words = []
        if boxes is not None:
            log.debug("detected {} words".format(len(boxes)))
            for box in boxes:
                # avoid submitting errors
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue

                pt1 = (box[0, 0], box[0, 1])
                pt2 = (box[1, 0], box[1, 1])
                pt3 = (box[2, 0], box[2, 1])
                pt4 = (box[3, 0], box[3, 1])
                word_img = dumpRotateImage(image, degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])), pt1, pt2, pt3, pt4)

                word = (self.recognizer.recognize(word_img))[self.text_direction]
                words.append(word)

        log.debug("completed recognition")
        return words, boxes



def main():
    logging.basicConfig(level=logging.DEBUG
        , format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.5
    fontColor = (0,69,255)
    lineType = 1

    def get_image_paths(image_path):
        types = ('.jpg', '.png', '.jpeg', '.JPG')

        if os.path.isfile(image_path) and image_path.endswith(types):
            return [image_path]

        files = []
        for t in types:
          files.extend(glob.glob(os.path.join(image_path, '*' + t)))

        return files

    parser = argparse.ArgumentParser()
    parser.add_argument('input_image_path', type=str)
    parser.add_argument('output_path', type=str)
    FLAGS = parser.parse_args()

    pipeline = AllWordsRecognizer()

    img_list = get_image_paths(FLAGS.input_image_path)
    logging.info('Found {} images'.format(len(img_list)))

    for img_file in img_list:
        img = cv2.imread(img_file)[:, :, ::-1]
        words, boxes=pipeline.get_all_words(img)

        if words:
            res_file = os.path.join(
                FLAGS.output_path,
                '{}.txt'.format(
                    os.path.basename(img_file).split('.')[0]))

            with open(res_file, 'w') as f:
                for idx, box in enumerate(boxes):
                    bottomLeftCornerOfText = (box[0, 0], box[0, 1])
                    cv2.putText(img[:, :, ::-1], words[idx],
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType)

                    f.write('{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{}\r\n'.format(
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],words[idx]
                    ))
                    cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)

        img_path = os.path.join(FLAGS.output_path, os.path.basename(img_file))
        cv2.imwrite(img_path, img[:, :, ::-1])
