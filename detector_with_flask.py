import argparse
import os
import logging
import json

from eyewitness.flask_server import BboxObjectDetectionFlaskWrapper
from eyewitness.config import BBOX
from eyewitness.detection_result_filter import FeedbackBboxDeNoiseFilter
from eyewitness.result_handler.db_writer import BboxPeeweeDbWriter
from peewee import SqliteDatabase

from naive_detector import YoloV3DetectorWrapper
from yolo import YOLO
from line_detection_result_handler import LineAnnotationSender
from facebook_detection_result_handler import FaceBookAnnoationSender

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


def image_url_handler(drawn_image_path):
    """if site_domain not set in env, will pass a pickchu image
    used for generate the detected image url for LineAnnotationSender
    """
    site_domain = os.environ.get('site_domain')
    if site_domain is None:
        return 'https://upload.wikimedia.org/wikipedia/en/a/a6/Pok%C3%A9mon_Pikachu_art.png'
    else:
        return '%s/%s' % (site_domain, drawn_image_path)


def raw_image_url_handler(drawn_image_path):
    """if site_domain not set in env, will pass a pickchu image
    used for generate the raw image url for LineAnnotationSender
    """
    site_domain = os.environ.get('site_domain')
    raw_image_path = drawn_image_path.replace('detected_image/', 'raw_image/')
    if site_domain is None:
        return 'https://upload.wikimedia.org/wikipedia/en/a/a6/Pok%C3%A9mon_Pikachu_art.png'
    else:
        return '%s/%s' % (site_domain, raw_image_path)


def line_detection_result_filter(detection_result):
    """
    used to check if sent notification or not
    """
    return any(i.label == 'person' for i in detection_result.detected_objects)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    args = parser.parse_args()
    detection_threshold = 0.7
    # object detector
    object_detector = YoloV3DetectorWrapper(args, threshold=detection_threshold)

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
            raw_image_url_handler=raw_image_url_handler,
            detection_result_filter=line_detection_result_filter,
            detection_method=BBOX,
            update_audience_period=10,
            database=database)
        result_handlers.append(line_annotation_sender)

    fb_user_email = os.environ.get('FACEBOOK_USER_EMAIL')
    if fb_user_email:
        fb_user_password = os.environ.get('FACEBOOK_USER_PASSWORD')
        fb_session_cookie_path = os.environ.get('FACEBOOK_SESSION_COOKIES_PATH')
        audience_id_str = os.environ.get('YOUR_USER_ID')
        audience_ids = set([i for i in audience_id_str.split(',') if i])
        with open(fb_session_cookie_path, 'r') as f:
            session_dict = json.load(f)

        facebook_annotation_sender = FaceBookAnnoationSender(
            audience_ids=audience_ids,
            user_email=fb_user_email,
            user_password=fb_user_password,
            session_dict=session_dict,
            image_url_handler=image_url_handler,
            detection_result_filter=line_detection_result_filter,
            detection_method=BBOX)
        result_handlers.append(facebook_annotation_sender)

    # denoise filter
    denoise_filters = []
    denoise_filter = FeedbackBboxDeNoiseFilter(
        database, detection_threshold=detection_threshold)
    denoise_filters.append(denoise_filter)

    flask_wrapper = BboxObjectDetectionFlaskWrapper(
        object_detector, bbox_sqlite_handler, result_handlers,
        database=database, drawn_image_dir=args.drawn_image_dir,
        detection_result_filters=denoise_filters)

    params = {'host': args.detector_host, 'port': args.detector_port, 'threaded': False}
    flask_wrapper.app.run(**params)
