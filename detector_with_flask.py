import argparse
import os
from collections import Counter

from eyewitness.flask_server import BboxObjectDetectionFlaskWrapper
from eyewitness.config import BBOX
from eyewitness.detection_utils import DetectionResult
from eyewitness.image_id import ImageId
from eyewitness.object_detector import ObjectDetector
from eyewitness.result_handler.db_writer import BboxPeeweeDbWriter
from eyewitness.result_handler.line_detection_result_handler import LineAnnotationSender
from peewee import SqliteDatabase
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

parser.add_argument(
    '--db_path', type=str, default='::memory::',
    help='the path used to store detection result records'
)

parser.add_argument(
    '--detector_host', type=str, default='localhost', help='the ip address of detector'
)
parser.add_argument(
    '--detector_port', type=int, default=5566, help='the port of detector port'
)
parser.add_argument(
    '--drawn_image_dir', type=str, default=None,
    help='the path used to store drawn images'
)


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

        detected_objs = Counter(i[4] for i in detected_objects)
        print("detected %s objects: %s" % (len(detected_objects), detected_objs))
        image_dict = {
            'image_id': image_id,
            'detected_objects': detected_objects,
        }
        detection_result = DetectionResult(image_dict)
        return detection_result


def image_url_handler(drawn_image_path):
    """if site_domain not set in env, will pass a pickchu image"""
    site_domain = os.environ.get('site_domain')
    if site_domain is None:
        return 'https://upload.wikimedia.org/wikipedia/en/a/a6/Pok%C3%A9mon_Pikachu_art.png'
    else:
        return '%s/%s' % (site_domain, drawn_image_path)


def line_detection_result_filter(detection_result):
    """
    used to check if sent notification or not
    """
    return any(i.label == 'person' for i in detection_result.detected_objects)


if __name__ == '__main__':
    args = parser.parse_args()

    # object detector
    object_detector = YoloV3DetectorWrapper(args)

    # detection result handlers
    result_handlers = []

    # update image_info drawn_image_path, insert detection result
    database = SqliteDatabase(args.db_path)
    bbox_sqlite_handler = BboxPeeweeDbWriter(database)
    result_handlers.append(bbox_sqlite_handler)

    # setup your line channel token and audience
    channel_access_token = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
    if channel_access_token:
        line_annotation_sender = LineAnnotationSender(
            channel_access_token=channel_access_token,
            image_url_handler=image_url_handler,
            detection_result_filter=line_detection_result_filter,
            detection_method=BBOX,
            update_audience_period=10,
            database=database)
        result_handlers.append(line_annotation_sender)

    flask_wrapper = BboxObjectDetectionFlaskWrapper(
        object_detector, bbox_sqlite_handler, result_handlers,
        database=database, drawn_image_dir=args.drawn_image_dir)

    params = {'host': args.detector_host, 'port': args.detector_port, }
    flask_wrapper.app.run(**params)
