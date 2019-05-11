import os
import urllib.request

import arrow
from celery import Celery
from eyewitness.image_id import ImageId
from eyewitness.image_utils import ImageHandler, Image
from bistiming import Stopwatch

from naive_detector import YoloV3DetectorWrapper
from yolo import YOLO


# system-wise setting, folder, rabbitmq url
RAW_IMAGE_FOLDER = os.environ.get('raw_image_fodler', 'raw_image')
DETECTED_IMAGE_FOLDER = os.environ.get('detected_image_folder', 'detected_image')
BROKER_URL = os.environ.get('broker_url', 'amqp://guest:guest@rabbitmq:5672')
celery = Celery('tasks', broker=BROKER_URL)


# model-wise setting
model_config = {
    'model': os.environ.get('model', YOLO.get_defaults("model_path")),
    'anchors': os.environ.get('anchors', YOLO.get_defaults("anchors_path")),
    'classes': os.environ.get('classes', YOLO.get_defaults("classes_path")),
    'gpu_num': os.environ.get('gpu_num', YOLO.get_defaults("gpu_num")),
}
threshold = os.environ.get('threshold', YOLO.get_defaults("threshold"))
# initialize a global detector first
GLOBAL_OBJECT_DETECTOR = YoloV3DetectorWrapper(model_config, threshold=threshold)


def generate_image_url(channel):
    return "https://upload.wikimedia.org/wikipedia/commons/2/25/5566_and_Daily_Air_B-55507_20050820.jpg"  # noqa


def generate_image(channel, timestamp, raw_image_path=None):
    image_id = ImageId(channel=channel, timestamp=timestamp, file_format='jpg')
    if not raw_image_path:
        raw_image_path = "%s/%s.jpg" % (RAW_IMAGE_FOLDER, str(image_id))
        # generate raw image
        urllib.request.urlretrieve(generate_image_url(channel), raw_image_path)
    return Image(image_id, raw_image_path=raw_image_path)


@celery.task
def detect_image(params):
    channel = params.get('channel', 'demo')
    timestamp = params.get('timestamp', arrow.now().timestamp)
    is_store_detected_image = params.get('is_store_detected_image', True)
    raw_image_path = params.get('raw_image_path')

    image_obj = generate_image(channel, timestamp, raw_image_path)

    with Stopwatch('Running inference on image {}...'.format(image_obj.raw_image_path)):
        detection_result = GLOBAL_OBJECT_DETECTOR.detect(image_obj)

    # TODO: add filter, detection result handlers

    if is_store_detected_image:
        ImageHandler.draw_bbox(image_obj.pil_image_obj, detection_result.detected_objects)
        ImageHandler.save(image_obj.pil_image_obj,
                          "%s/%s.jpg" % (DETECTED_IMAGE_FOLDER, str(image_obj.image_id)))
