import logging
import random
import typing
from abc import ABC, abstractmethod
from copy import deepcopy

from ..common import DatasetManifest, DatasetTypes, ImageDataManifest
from ..common.dataset.base_dataset import BaseDataset
from ..common.dataset.vision_dataset import LocalFolderCacheDecorator, VisionDataset
from ..image_classification.manifest import ImageClassificationLabelManifest
from .manifest import ImageObjectDetectionLabelManifest

logger = logging.getLogger(__name__)


class DetectionAsClassificationBaseDataset(BaseDataset, ABC):
    def __init__(self, detection_dataset: VisionDataset, dataset_type: DatasetTypes):
        """
        Args:
            detection_dataset: the detection dataset where images are cropped as classification samples
        """

        if detection_dataset is None:
            raise ValueError

        if detection_dataset.dataset_info.type != DatasetTypes.IMAGE_OBJECT_DETECTION:
            raise ValueError

        if dataset_type not in [DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL]:
            raise ValueError

        dataset_info = deepcopy(detection_dataset.dataset_info)
        dataset_info.type = dataset_type
        super().__init__(dataset_info)

        self._dataset = detection_dataset

    def close(self):
        self._dataset.close()

    @property
    def categories(self):
        return self._dataset.categories

    @abstractmethod
    def generate_manifest(self, **kwargs):
        pass


class DetectionAsClassificationIgnoreBoxesDataset(DetectionAsClassificationBaseDataset):
    """
    Consume a detection dataset as a multilabel classification dataset by simply ignoring the boxes. Duplicate classes for an image will be merged into one, i.e., whether one image possesses 1 bbox
    of category 1 or 100 bboxes of category 1 does not matter, after conversion
    """

    def __init__(self, detection_dataset: VisionDataset):
        super(DetectionAsClassificationIgnoreBoxesDataset, self).__init__(detection_dataset, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)

    def __len__(self):
        return len(self._dataset)

    def _get_single_item(self, index):
        img, labels, idx_str = self._dataset[index]
        labels = DetectionAsClassificationIgnoreBoxesDataset._od_to_ic_labels(labels)
        return img, labels, idx_str

    def generate_manifest(self, **kwargs):
        """
        Generate dataset manifest for the multilabel classification dataset converted from detection dataset by ignoring the bbox. Manifest will re-use the existing image paths
        """

        images = []
        for img in self._dataset.dataset_manifest.images:
            labels = DetectionAsClassificationIgnoreBoxesDataset._od_to_ic_labels(img.labels)
            ic_img = ImageDataManifest(len(images) + 1, img.img_path, img.width, img.height, labels)
            images.append(ic_img)
        return DatasetManifest(images, self._dataset.categories, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, self._dataset.dataset_manifest.additional_info)

    @staticmethod
    def _od_to_ic_labels(labels: typing.List[ImageObjectDetectionLabelManifest]):
        category_ids = sorted(list(set([label.category_id for label in labels])))
        return [ImageClassificationLabelManifest(x) for x in category_ids]


class DetectionAsClassificationByCroppingDataset(DetectionAsClassificationBaseDataset):
    """
    Consume detection dataset as a classification dataset, i.e., sample from this dataset is a crop wrt a bbox in the detection dataset.

    When box_aug_params is provided, different crops with randomness will be generated for the same bbox
    """

    def __init__(self, detection_dataset: VisionDataset, box_aug_params: dict = None):
        """
        Args:
            detection_dataset: the detection dataset where images are cropped as classification samples
            box_aug_params (dict): params controlling box crop augmentation,
                'zoom_ratio_bounds': the lower/upper bound of box zoom ratio wrt box width and height, e.g., (0.3, 1.5)
                'shift_relative_bounds': lower/upper bounds of relative ratio wrt box width and height that a box can shift, e.g., (-0.3, 0.1)
                'rnd_seed' [optional]: rnd seed used for box crop zoom and shift, default being 0
        """
        super().__init__(detection_dataset, DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)

        self._n_booxes = 0
        self._box_abs_id_to_img_rel_id = {}
        for img_id, x in enumerate(self._dataset.dataset_manifest.images):
            for i in range(len(x.labels)):
                self._box_abs_id_to_img_rel_id[self._n_booxes] = (img_id, i)
                self._n_booxes += 1
        self._box_aug_params = box_aug_params

        self._box_aug_rnd = random.Random(self._box_aug_params.get('rnd_seed', 0)) if box_aug_params else None
        self._box_pick_rnd = random.Random(0)

    def __len__(self):
        return self._n_booxes

    def _get_single_item(self, index):
        img_idx, box_rel_idx = self._box_abs_id_to_img_rel_id[index]

        img, boxes, _ = self._dataset[img_idx]
        c_id, left, t, r, b = boxes[box_rel_idx].label_data
        if self._dataset.coordinates == 'relative':
            w, h = img.size
            left, t, r, b = left * w, t * h, r * w, b * h

        box_img = DetectionAsClassificationByCroppingDataset.crop(img, left, t, r, b, self._box_aug_params, self._box_aug_rnd)
        return box_img, [ImageClassificationLabelManifest(c_id)], str(index)

    @staticmethod
    def crop(img, left, t, r, b, aug_params=None, rnd: random.Random = None):
        if aug_params:
            if not rnd:
                raise ValueError
            if 'zoom_ratio_bounds' in aug_params:
                ratio_lower_b, ratio_upper_b = aug_params['zoom_ratio_bounds']
                left, t, r, b = BoxAlteration.zoom_box(left, t, r, b, img.size[0], img.size[1], ratio_lower_b, ratio_upper_b, rnd)

            if 'shift_relative_bounds' in aug_params:
                relative_lower_b, relative_upper_b = aug_params['shift_relative_bounds']
                left, t, r, b = BoxAlteration.shift_box(left, t, r, b, img.size[0], img.size[1], relative_lower_b, relative_upper_b, rnd)

        crop_img = img.crop((left, t, r, b))
        crop_img.format = img.format

        return crop_img

    def generate_manifest(self, **kwargs):
        """
        Generate dataset manifest for the multiclass classification dataset converted from detection dataset by cropping bboxes as classification samples.
        Crops will be saved into 'dir' for generating the manifest
        Args:
            'dir'(str): directory where cropped images will be saved
            'n_copies'(int): number of image copies generated for each bbox
        """

        local_cache_params = {'dir': kwargs.get('dir', f'{self.dataset_info.name}-cropped-ic'), 'n_copies': kwargs.get('n_copies')}
        cache_decor = LocalFolderCacheDecorator(self, local_cache_params)
        return cache_decor.generate_manifest()


class BoxAlteration:
    @staticmethod
    def _stay_in_range(val, low, up):
        return int(min(max(val, low), up))

    @staticmethod
    def shift_box(left, t, r, b, img_w, img_h, relative_lower_b, relative_upper_b, rnd: random.Random):
        # level = logging.DEBUG
        # logger.log(level, f'old box {left}, {t}, {r}, {b}, out of ({img_w}, {img_h})')
        box_w = r - left
        box_h = b - t
        hor_shift = rnd.uniform(relative_lower_b, relative_upper_b) * box_w
        ver_shift = rnd.uniform(relative_lower_b, relative_upper_b) * box_h
        left = BoxAlteration._stay_in_range(left + hor_shift, 0, img_w)
        t = BoxAlteration._stay_in_range(t + ver_shift, 0, img_h)
        r = BoxAlteration._stay_in_range(r + hor_shift, 0, img_w)
        b = BoxAlteration._stay_in_range(b + ver_shift, 0, img_h)
        # logger.log(level, f'[shift_box] new box {left}, {t}, {r}, {b}, with {hor_shift}, {ver_shift}, out of ({img_w}, {img_h})')

        return left, t, r, b

    @staticmethod
    def zoom_box(left, t, r, b, img_w, img_h, ratio_lower_b, ratio_upper_b, rnd: random.Random):
        # level = logging.DEBUG
        # logger.log(level, f'old box {left}, {t}, {r}, {b}, out of ({img_w}, {img_h})')
        w_ratio = rnd.uniform(ratio_lower_b, ratio_upper_b)
        h_ratio = rnd.uniform(ratio_lower_b, ratio_upper_b)
        box_w = r - left
        box_h = b - t
        new_box_w = box_w * w_ratio
        new_box_h = box_h * h_ratio
        # logger.log(level, f'w h change: {box_w} {box_h} => {new_box_w} {new_box_h}')
        left = BoxAlteration._stay_in_range(left - (new_box_w - box_w) / 2, 0, img_w)
        t = BoxAlteration._stay_in_range(t - (new_box_h - box_h) / 2, 0, img_h)
        r = BoxAlteration._stay_in_range(r + (new_box_w - box_w) / 2, left, img_w)
        b = BoxAlteration._stay_in_range(b + (new_box_h - box_h) / 2, 0, img_h)
        # logger.log(level, f'[zoom_box] new box {left}, {t}, {r}, {b}, with {w_ratio}, {h_ratio}, out of ({img_w}, {img_h})')

        return left, t, r, b
