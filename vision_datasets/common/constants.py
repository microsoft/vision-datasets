from enum import Enum


class DatasetTypes(Enum):
    MULTITASK = 1
    IMAGE_CLASSIFICATION_MULTILABEL = 2
    IMAGE_CLASSIFICATION_MULTICLASS = 3
    IMAGE_OBJECT_DETECTION = 4
    IMAGE_CAPTION = 5
    IMAGE_TEXT_MATCHING = 6
    IMAGE_MATTING = 7
    IMAGE_REGRESSION = 8
    TEXT_2_IMAGE_RETRIEVAL = 9


class Usages(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class AnnotationFormats(Enum):
    COCO = 1
    IRIS = 2


class BBoxFormat(Enum):
    LTRB = 1
    LTWH = 2
