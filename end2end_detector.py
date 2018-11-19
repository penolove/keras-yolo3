import argparse

import arrow
import cv2
import numpy as np
from eyewitness.detection_utils import DetectionResult
from eyewitness.image_id import ImageId
from eyewitness.object_detector import ObjectDetector
from eyewitness.image_utils import (ImageProducer, swap_channel_rgb_bgr)
from eyewitness.config import IN_MEMORY
from PIL import Image

from yolo import YOLO


# class YOLO defines the default value, so suppress any default here
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument(
    '--model', type=str,
    help='path to model weight file, default: ' + YOLO.get_defaults("model_path")
)

parser.add_argument(
    '--anchors', type=str,
    help='path to anchor definitions, default: ' + YOLO.get_defaults("anchors_path")
)

parser.add_argument(
    '--classes', type=str,
    help='path to class definitions, default: ' + YOLO.get_defaults("classes_path")
)

parser.add_argument(
    '--gpu_num', type=int,
    help='Number of GPU to use, default: ' + str(YOLO.get_defaults("gpu_num"))
)


class InMemoryImageProducer(ImageProducer):
    def __init__(self, video_path, interval_ms=1000):
        self.vid = cv2.VideoCapture(video_path)
        self.interval_ms = interval_ms
        if not self.vid.isOpened():
            raise IOError("Couldn't open webcam or video")

    def produce_method(self):
        return IN_MEMORY

    def produce_image(self) -> np.array:
        _, frame = self.vid.read()
        yield Image.fromarray(frame)
        cv2.waitKey(self.interval_ms)
        

class YoloV3DetectorWrapper(ObjectDetector):
    def __init__(self, model_config, threshold=0.5):
        self.core_model = YOLO(**vars(model_config))
        self.threshold = threshold

    def detect(self, image: Image, image_id: ImageId) -> DetectionResult:
        (out_boxes, out_scores, out_classes) = self.core_model.predict(image)
        detected_objects = []
        for bbox, score, label_class in zip(out_boxes, out_scores, out_classes):
            label = self.core_model.class_names[label_class]
            y1, x1, y2, x2 = bbox
            if score > self.threshold:
                detected_objects.append([x1, y1, x2, y2, label, score, ''])

        image_dict = {
            'image_id': image_id,
            'detected_objects': detected_objects,
        }
        detection_result = DetectionResult(image_dict)
        return detection_result


if __name__ == '__main__':
    model_config = parser.parse_args()
    object_detector = YoloV3DetectorWrapper(model_config)
    image_producer = InMemoryImageProducer(0)  # image producer from webcam

    while True:
        image = swap_channel_rgb_bgr(image_producer.produce_image())
        image_id = ImageId(channel='demo', timestamp=arrow.now().timestamp, file_format='jpg')
        detection_result = object_detector.detect(image, image_id)
        # ImageHandler.draw_bbox(image, detection_result.detected_objects)
        # ImageHandler.save(image, "demo/drawn_image.jpg")