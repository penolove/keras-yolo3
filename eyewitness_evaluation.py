import argparse

from eyewitness.dataset_util import BboxDataSet
from eyewitness.evaluation import BboxMAPEvaluator

from naive_detector import YoloV3DetectorWrapper
from yolo import YOLO

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
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


if __name__ == '__main__':
    args = parser.parse_args()
    dataset_folder = 'VOC2007'
    dataset_VOC_2007 = BboxDataSet(dataset_folder, 'VOC2007')
    object_detector = YoloV3DetectorWrapper(args, threshold=0.0)
    bbox_map_evaluator = BboxMAPEvaluator(test_set_only=False)
    # which will lead to ~0.73
    print(bbox_map_evaluator.evaluate(object_detector, dataset_VOC_2007))
