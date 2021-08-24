class DatasetTypes:
    IC_MULTILABEL = 'classification_multilabel'
    IC_MULTICLASS = 'classification_multiclass'
    OD = 'object_detection'
    MULTITASK = 'multitask'

    VALID_TYPES = [IC_MULTILABEL, IC_MULTICLASS, OD, MULTITASK]

    @staticmethod
    def is_classification(dataset_type):
        return dataset_type.startswith('classification')


class Usages:
    TRAIN_PURPOSE = 'train'
    VAL_PURPOSE = 'val'
    TEST_PURPOSE = 'test'


class Formats:
    IRIS = 'iris'
    COCO = 'coco'
