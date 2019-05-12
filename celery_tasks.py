import os
import urllib.request

import arrow
from celery import Celery
from eyewitness.config import (
    BBOX,
    RAW_IMAGE_PATH,
)
from eyewitness.detection_result_filter import FeedbackBboxDeNoiseFilter
from eyewitness.image_id import ImageId
from eyewitness.image_utils import ImageHandler, Image
from eyewitness.result_handler.db_writer import BboxPeeweeDbWriter
from eyewitness.result_handler.line_detection_result_handler import LineAnnotationSender
from peewee import SqliteDatabase
from bistiming import Stopwatch

from naive_detector import YoloV3DetectorWrapper
from detector_with_flask import (
    raw_image_url_handler, image_url_handler, line_detection_result_filter)

from yolo import YOLO

# system-wise setting, folder, rabbitmq url, db connection, line chatbot
RAW_IMAGE_FOLDER = os.environ.get('raw_image_dir', 'raw_image')
DRAWN_IMAGE_DIR = os.environ.get('drawn_image_dir', 'detected_image')
SQLITE_DB_PATH = os.environ.get('db_path', 'db_folder/example.sqlite')
BROKER_URL = os.environ.get('broker_url', 'amqp://guest:guest@rabbitmq:5672')
CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
celery = Celery('tasks', broker=BROKER_URL)

# model-wise setting
model_config = {
    'model': os.environ.get('model', YOLO.get_defaults("model_path")),
    'anchors': os.environ.get('anchors', YOLO.get_defaults("anchors_path")),
    'classes': os.environ.get('classes', YOLO.get_defaults("classes_path")),
    'gpu_num': os.environ.get('gpu_num', YOLO.get_defaults("gpu_num")),
}
threshold = os.environ.get('threshold', 0.7)
# initialize a global detector first
GLOBAL_OBJECT_DETECTOR = YoloV3DetectorWrapper(model_config, threshold=threshold)

# detection result handler
DETECTION_RESULT_FILTERS = []
DETECTION_RESULT_HANDLERS = []
DATABASE = SqliteDatabase(SQLITE_DB_PATH)
DETECTION_RESULT_HANDLERS.append(BboxPeeweeDbWriter(DATABASE))
# setup your line channel token and audience
if CHANNEL_ACCESS_TOKEN:
    line_annotation_sender = LineAnnotationSender(
        channel_access_token=CHANNEL_ACCESS_TOKEN,
        image_url_handler=image_url_handler,
        raw_image_url_handler=raw_image_url_handler,
        detection_result_filter=line_detection_result_filter,
        detection_method=BBOX,
        update_audience_period=10,
        database=DATABASE)
    DETECTION_RESULT_HANDLERS.append(line_annotation_sender)

# detection result filter
DETECTION_RESULT_FILTERS.append(FeedbackBboxDeNoiseFilter(
    DATABASE, detection_threshold=threshold))


def generate_image_url(channel):
    return "https://upload.wikimedia.org/wikipedia/commons/2/25/5566_and_Daily_Air_B-55507_20050820.jpg"  # noqa


def generate_image(channel, timestamp, image_register=None, raw_image_path=None):
    image_id = ImageId(channel=channel, timestamp=timestamp, file_format='jpg')
    if not raw_image_path:
        raw_image_path = "%s/%s.jpg" % (RAW_IMAGE_FOLDER, str(image_id))
        # generate raw image
        urllib.request.urlretrieve(generate_image_url(channel), raw_image_path)
    if image_register:
        image_register.register_image(image_id, {RAW_IMAGE_PATH: raw_image_path})
    return Image(image_id, raw_image_path=raw_image_path)


@celery.task
def detect_image(params):
    channel = params.get('channel', 'demo')
    timestamp = params.get('timestamp', arrow.now().timestamp)
    is_store_detected_image = params.get('is_store_detected_image', True)
    raw_image_path = params.get('raw_image_path')

    image_obj = generate_image(
        channel, timestamp, BboxPeeweeDbWriter, raw_image_path=raw_image_path)

    with Stopwatch('Running inference on image {}...'.format(image_obj.raw_image_path)):
        detection_result = GLOBAL_OBJECT_DETECTOR.detect(image_obj)

    for detection_result_filter in DETECTION_RESULT_FILTERS:
        detection_result = detection_result_filter.apply(detection_result)

    if is_store_detected_image and len(detection_result.detected_objects) > 0:
        ImageHandler.draw_bbox(image_obj.pil_image_obj, detection_result.detected_objects)
        ImageHandler.save(image_obj.pil_image_obj, "%s/%s.jpg" % (
            DRAWN_IMAGE_DIR, str(image_obj.image_id)))

    for detection_result_handler in DETECTION_RESULT_HANDLERS:
        detection_result_handler.handle(detection_result)
