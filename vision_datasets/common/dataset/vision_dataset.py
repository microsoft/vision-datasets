import copy
import logging
import os.path
import pathlib
import typing

from PIL import Image, JpegImagePlugin
from tqdm import tqdm

from ..constants import DatasetTypes
from ..data_reader import FileReader, PILImageLoader
from ..dataset_info import BaseDatasetInfo
from ..data_manifest import DatasetManifest, ImageDataManifest
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class VisionDataset(BaseDataset):
    """Dataset class that accesses data from dataset manifest.

    """

    def __init__(self, dataset_info: BaseDatasetInfo, dataset_manifest: DatasetManifest, coordinates='relative', dataset_resources=None):
        """

        Args:
            dataset_info (BaseDatasetInfo): dataset info, containing high level information about the dataset, such as name, type, description, etc
            dataset_manifest (DatasetManifest): dataset manifest containing meta data such as image paths, annotations, etc
            coordinates (str): 'relative' or 'absolute', indicating the desired format of the bboxes returned. Works for detection dataset only.
                    This params will be refactored out later as it is OD-specific.
            dataset_resources (str): disposable resources associated with this dataset
        """

        if dataset_manifest is None:
            raise ValueError

        if coordinates not in ['relative', 'absolute']:
            raise ValueError

        super().__init__(dataset_info)

        self.dataset_manifest = dataset_manifest
        self.coordinates = coordinates
        self._file_reader = FileReader()
        self.dataset_resources = dataset_resources

    @property
    def categories(self):
        return self.dataset_manifest.categories

    def get_targets(self, index):
        image_manifest: ImageDataManifest = self.dataset_manifest.images[index]
        targets = image_manifest.labels
        w, h = image_manifest.width, image_manifest.height

        def load_image():
            return self._load_image(image_manifest.img_path)

        targets = VisionDataset._convert_box_to_relative_if_od(image_manifest.labels, w, h, load_image, self.dataset_info)

        return targets

    def __len__(self):
        return len(self.dataset_manifest.images)

    def _get_single_item(self, index):
        image_manifest: ImageDataManifest = self.dataset_manifest.images[index]
        image = self._load_image(image_manifest.img_path)
        target = image_manifest.labels
        if self.coordinates == 'relative':
            w, h = image.size
            target = VisionDataset._convert_box_to_relative_if_od(image_manifest.labels, w, h, None, self.dataset_info)

        return image, target, str(index)

    def close(self):
        self._file_reader.close()

    def _load_image(self, filepath):
        try:
            with self._file_reader.open(filepath, 'rb') as f:
                img = PILImageLoader.load_from_stream(f)
                logger.debug(f'Loaded image from path: {filepath}')
                return img
        except Exception:
            logger.exception(f'Failed to load an image with path: {filepath}')
            raise

    @staticmethod
    def _convert_box_to_relative_if_od(target: typing.Union[typing.List, dict], img_w, img_h, load_image, dataset_info):
        # Convert absolute coordinates to relative coordinates.
        # Example: for image with size (200, 200), (1, 100, 100, 200, 200) => (1, 0.5, 0.5, 1.0, 1.0)
        if dataset_info.type == DatasetTypes.MULTITASK:
            return {task_name: VisionDataset._convert_box_to_relative_if_od(task_target, img_w, img_h, load_image, dataset_info.sub_task_infos[task_name]) for task_name, task_target in target.items()}

        if dataset_info.type == DatasetTypes.IMAGE_OBJECT_DETECTION:
            relative_target = copy.deepcopy(target)
            if not img_w or not img_h:
                img_w, img_h = load_image().size

            for t in relative_target:
                label = t.label_data
                t.label_data = [label[0], label[1] / img_w, label[2] / img_h, label[3] / img_w, label[4] / img_h]
            return relative_target

        return target


class LocalFolderCacheDecorator(BaseDataset):
    """
    Decorate a dataset by caching data in a local folder, in local_cache_params['dir'].

    """

    def __init__(self, dataset: BaseDataset, local_cache_params: dict):
        """
        Args:
            dataset: dataset that requires cache
            local_cache_params(dict): params controlling local cache for image access:
                'dir': local dir for caching crops, it will be auto-created if not exist
                [optional] 'n_copies': default being 1. if n_copies is greater than 1, then multiple copies will be cached and dataset will be n_copies times bigger
        """

        if dataset is None:
            raise ValueError
        if not local_cache_params or not local_cache_params.get('dir'):
            raise ValueError

        local_cache_params['n_copies'] = local_cache_params.get('n_copies', 1)

        if local_cache_params['n_copies'] < 1:
            raise ValueError('n_copies must be equal or greater than 1.')

        super().__init__(dataset.dataset_info)

        self._dataset = dataset
        self._local_cache_params = local_cache_params
        if not os.path.exists(self._local_cache_params['dir']):
            os.makedirs(self._local_cache_params['dir'])

        self._local_dir = pathlib.Path(self._local_cache_params['dir'])
        self._annotations = {}
        self._paths = {}

    @property
    def categories(self):
        return self._dataset.categories

    def __len__(self):
        return len(self._dataset) * self._local_cache_params['n_copies']

    def _get_single_item(self, index):
        annotations = self._annotations.get(index)
        if annotations:
            return Image.open(self._paths[index]), annotations, str(index)

        idx_in_epoch = index % len(self._dataset)
        img, annotations, _ = self._dataset[idx_in_epoch]
        local_img_path = self._construct_local_image_path(index, img.format)
        self._save_image_matching_quality(img, local_img_path)
        self._annotations[index] = annotations
        self._paths[index] = local_img_path

        return img, annotations, str(index)

    def _construct_local_image_path(self, img_idx, img_format):
        return self._local_dir / f'{img_idx}.{img_format}'

    def _save_image_matching_quality(self, img, fp):
        """
        Save the image with mathcing qulaity, try not to compress
        https://stackoverflow.com/a/56675440/2612496
        """
        frmt = img.format

        if frmt == 'JPEG':
            quantization = getattr(img, 'quantization', None)
            subsampling = JpegImagePlugin.get_sampling(img)
            quality = 100 if quantization is None else -1
            img.save(fp, format=frmt, subsampling=subsampling, qtables=quantization, quality=quality)
        else:
            img.save(fp, format=frmt, quality=100)

    def generate_manifest(self):
        """
        Generate dataset manifest for the cached dataset.
        """

        images = []
        for idx in tqdm(range(len(self)), desc='Generating manifest...'):
            img, labels, _ = self._get_single_item(idx)  # make sure
            width, height = img.size
            image = ImageDataManifest(len(images) + 1, str(self._paths[idx].as_posix()), width, height, labels)
            images.append(image)

        manifest = getattr(self._dataset, "dataset_manifest", None)
        additional_info = None if manifest is None else manifest.additional_info
        return DatasetManifest(images, self.categories, self._dataset.dataset_info.type, additional_info)

    def close(self):
        self._dataset.close()
