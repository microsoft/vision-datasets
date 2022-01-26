import logging
import os.path
import pathlib
from copy import deepcopy
import random
from PIL import Image

from .base_dataset import BaseDataset
from .constants import DatasetTypes
from .dataset_info import BaseDatasetInfo
from .data_manifest import DatasetManifest
from .image_loader import PILImageLoader
from .util import FileReader

logger = logging.getLogger(__name__)


class ManifestDataset(BaseDataset):
    """Dataset class that accesses data from dataset manifest
    """

    def __init__(self, dataset_info: BaseDatasetInfo, dataset_manifest: DatasetManifest, coordinates='relative', dataset_resources=None):
        """

        Args:
            dataset_info (BaseDatasetInfo): dataset info, containing high level information about the dataset, such as name, type, description, etc
            dataset_manifest (DatasetManifest): dataset manifest containing meta data such as image paths, annotations, etc
            coordinates (str): 'relative' or 'absolute', indicating the desired format of the bboxes returned.
            dataset_resources (str): disposable resources associated with this dataset
        """
        assert dataset_manifest
        assert coordinates in ['relative', 'absolute']

        super().__init__(dataset_info)

        self.dataset_manifest = dataset_manifest
        self.coordinates = coordinates
        self._file_reader = FileReader()
        self.dataset_resources = dataset_resources

    @property
    def labels(self):
        return self.dataset_manifest.labelmap

    def __len__(self):
        return len(self.dataset_manifest.images)

    def _get_single_item(self, index):
        image_manifest = self.dataset_manifest.images[index]
        image = self._load_image(image_manifest.img_path)
        target = image_manifest.labels
        if self.coordinates == 'relative':
            w, h = image.size
            target = ManifestDataset._box_convert_to_relative(image_manifest.labels, w, h, self.dataset_info)

        return image, target, str(index)

    def close(self):
        self._file_reader.close()

    def _load_image(self, filepath):
        full_path = filepath.replace('\\', '/')
        try:
            with self._file_reader.open(full_path, 'rb') as f:
                return PILImageLoader.load_from_stream(f)
        except Exception:
            logger.exception(f'Failed to load an image with path: {full_path}')
            raise

    @staticmethod
    def _box_convert_to_relative(target, w, h, dataset_info):
        # Convert absolute coordinates to relative coordinates.
        # Example: for image with size (200, 200), (1, 100, 100, 200, 200) => (1, 0.5, 0.5, 1.0, 1.0)
        if dataset_info.type == DatasetTypes.MULTITASK:
            return {task_name: ManifestDataset._box_convert_to_relative(task_target, w, h, dataset_info.sub_task_infos[task_name]) for task_name, task_target in target.items()}
        if dataset_info.type == DatasetTypes.OD:
            return [[t[0], t[1] / w, t[2] / h, t[3] / w, t[4] / h] for t in target]

        return target


class DetectionAsClassificationDataset(BaseDataset):
    """
    Consume detection dataset as a classification dataset, i.e., sample from this dataset is a crop wrt a bbox in the detection dataset.
    """

    def __init__(self, detection_dataset: ManifestDataset, box_aug_params: dict = None, local_cache_params: dict = None):
        """
        Args:
            detection_dataset: the detection dataset where images are cropped as classification samples
            box_aug_params (dict): params controlling box crop augmentation,
                'zoom_ratio_bounds': the lower/upper bound of box zoom ratio wrt box width and height, e.g., (-0.3, 0.1)
                'shift_relative_bounds': lower/upper bounds of relative ratio wrt box width and height that a box can shift, e.g., (-0.3, 0.1)
                'rnd_seed': rnd seed used for box crop zoom and shift
            local_cache_params(dict): params controlling local cache for crop access:
                'dir': local dir for caching crops, it will be auto-created if not exist
                'n_max_copies': max number of crops cached for each bbox
        """
        assert detection_dataset
        assert detection_dataset.dataset_info.type == DatasetTypes.OD

        dataset_info = deepcopy(detection_dataset.dataset_info)
        dataset_info.type = DatasetTypes.IC_MULTICLASS
        super().__init__(dataset_info)

        self._dataset = detection_dataset
        self._n_booxes = 0
        self._box_abs_id_to_img_rel_id = {}
        for img_id, x in enumerate(self._dataset):
            boxes = x[1]
            for i in range(len(boxes)):
                self._box_abs_id_to_img_rel_id[self._n_booxes] = (img_id, i)
                self._n_booxes += 1
        self._box_aug_params = box_aug_params

        self._box_aug_rnd = random.Random(self._box_aug_params['rnd_seed']) if box_aug_params else None
        self._box_pick_rnd = random.Random(0)
        self._local_cache_params = local_cache_params
        if self._local_cache_params and not os.path.exists(self._local_cache_params['dir']):
            os.makedirs(self._local_cache_params['dir'])

    @property
    def labels(self):
        return self._dataset.labels

    def __len__(self):
        return self._n_booxes

    def _get_single_item(self, index):
        local_img_path = None
        img_idx, box_rel_idx = self._box_abs_id_to_img_rel_id[index]
        if self._local_cache_params:
            box_copy_idx = self._box_pick_rnd.randint(0, self._local_cache_params["max_n_copies"] - 1)
            box_img_id = f'{index}-{box_copy_idx}' if self._box_aug_params else str(index)
            local_img_path = pathlib.Path(self._local_cache_params['dir']) / box_img_id
            if os.path.exists(local_img_path):
                logger.log(logging.DEBUG, f'Found local cache for crop {index}! {box_copy_idx}')
                c_id = self._dataset.dataset_manifest.images[img_idx].labels[box_rel_idx][0]
                return Image.open(local_img_path), [c_id], str(index)

        img, boxes, _ = self._dataset[img_idx]
        c_id, left, t, r, b = boxes[box_rel_idx]
        if self._dataset.coordinates == 'relative':
            w, h = img.size
            left *= w
            t *= h
            r *= w
            b *= h

        box_img = DetectionAsClassificationDataset.crop(img, left, t, r, b, self._box_aug_params, self._box_aug_rnd)
        if local_img_path:
            box_img.save(local_img_path, box_img.format)
        return box_img, [c_id], str(index)

    def close(self):
        self._dataset.close()

    @staticmethod
    def crop(img, left, t, r, b, aug_params=None, rnd: random.Random = None):
        if aug_params:
            assert rnd
            if 'zoom_ratio_bounds' in aug_params:
                ratio_lower_b, ratio_upper_b = aug_params['zoom_ratio_bounds']
                left, t, r, b = BoxAlteration.zoom_box(left, t, r, b, img.size[0], img.size[1], ratio_lower_b, ratio_upper_b, rnd)

            if 'shift_relative_bounds' in aug_params:
                relative_lower_b, relative_upper_b = aug_params['shift_relative_bounds']
                left, t, r, b = BoxAlteration.shift_box(left, t, r, b, img.size[0], img.size[1], relative_lower_b, relative_upper_b, rnd)

        crop_img = img.crop((left, t, r, b))
        crop_img.format = img.format

        return crop_img


class BoxAlteration:
    @staticmethod
    def shift_box(left, t, r, b, img_w, img_h, relative_lower_b, relative_upper_b, rnd: random.Random):
        level = logging.DEBUG
        logger.log(level, f'old box {left}, {t}, {r}, {b}, out of ({img_w}, {img_h})')
        box_w = r - left
        box_h = t - b
        hor_shift = rnd.uniform(relative_lower_b, relative_upper_b) * box_w
        ver_shift = rnd.uniform(relative_lower_b, relative_upper_b) * box_h
        left = min((left + hor_shift), img_w)
        t = min(t + ver_shift, img_h)
        r = min((r + hor_shift), img_w)
        b = min(b + ver_shift, img_h)
        logger.log(level, f'[shift_box] new box {left}, {t}, {r}, {b}, with {hor_shift}, {ver_shift}, out of ({img_w}, {img_h})')

        return left, t, r, b

    @staticmethod
    def zoom_box(left, t, r, b, img_w, img_h, ratio_lower_b, ratio_upper_b, rnd: random.Random):
        level = logging.DEBUG
        logger.log(level, f'old box {left}, {t}, {r}, {b}, out of ({img_w}, {img_h})')
        w_ratio = rnd.uniform(ratio_lower_b, ratio_upper_b)
        h_ratio = rnd.uniform(ratio_lower_b, ratio_upper_b)
        box_w = r - left
        box_h = b - t
        new_box_w = box_w * w_ratio
        new_box_h = box_h * h_ratio
        logger.log(level, f'w h change: {box_w} {box_h} => {new_box_w} {new_box_h}')
        left = max(left - (new_box_w - box_w) / 2, 0)
        t = max(t - (new_box_h - box_h) / 2, 0)
        r = min(r + (new_box_w - box_w) / 2, img_w)
        b = min(b + (new_box_h - box_h) / 2, img_h)
        logger.log(level, f'[zoom_box] new box {left}, {t}, {r}, {b}, with {w_ratio}, {h_ratio}, out of ({img_w}, {img_h})')

        return left, t, r, b
