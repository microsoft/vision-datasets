import random
from copy import deepcopy

from ..common import DatasetTypes
from .base_dataset import BaseDataset
from ..data_tasks.image_text_matching.manifest import ImageTextMatchingLabelManifest


class VisionAsImageTextDataset(BaseDataset):
    """
    Consume traditional vision datasets of type
    [DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, DatasetTypes.IMAGE_OBJECT_DETECTION], as DatasetTypes.IMAGE_TEXT_MATCHING dataset.
    For a certain image, negative image-text pairs are generated from the labels that this image does not possess.
    """

    def __init__(self, dataset: BaseDataset, neg_to_pos_ratio=0, text_aug=None, rnd_seed=0):
        """
        Args:
            dataset: dataset of expected type
            neg_to_pos_ratio: ratio of negative against positive image text pairs
            text_aug: a func that augments a string, i.e., a class name, e.g. dog => a photo of dog
            rnd_seed: random seed for choosing negative class names for negative image text pairs
        """
        assert dataset is not None
        assert dataset.dataset_info.type in [DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, DatasetTypes.IMAGE_OBJECT_DETECTION]
        assert neg_to_pos_ratio >= 0
        dataset_info = deepcopy(dataset.dataset_info)
        dataset_info.type = DatasetTypes.IMAGE_TEXT_MATCHING

        super().__init__(dataset_info)
        self._dataset = dataset
        self._negative_pair_ratio = neg_to_pos_ratio
        self._text_aug = text_aug or (lambda x: x)
        self._rand = random.Random(rnd_seed)

    @property
    def labels(self):
        return None

    def __len__(self):
        return len(self._dataset)

    def _get_single_item(self, index):
        img, target, _ = self._dataset[index]
        pos_class_indices = [x.category_id for x in target]
        pos_class_names = [self._dataset.labels[x].name for x in pos_class_indices]
        labels = [ImageTextMatchingLabelManifest((self._text_aug(class_name), 1)) for class_name in pos_class_names]
        if self._negative_pair_ratio > 0:
            neg_class_indices = set(range(len(self._dataset.labels))) - set(pos_class_indices)
            neg_class_names = [self._dataset.labels[x].name for x in neg_class_indices]
            if neg_class_names:
                down_sample_ratio = self._negative_pair_ratio * len(pos_class_names) / len(neg_class_names)
                if down_sample_ratio < 1:
                    neg_class_names = [ncn for ncn in neg_class_names if self._rand.random() < down_sample_ratio]

            neg_labels = [ImageTextMatchingLabelManifest((self._text_aug(class_name), 0)) for class_name in neg_class_names]
            labels += neg_labels
        return img, labels, str(index)

    def close(self):
        self._dataset.close()