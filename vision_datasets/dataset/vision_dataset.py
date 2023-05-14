import copy
import logging
import os.path
import pathlib
import typing

from PIL import Image, JpegImagePlugin
from tqdm import tqdm

from ..common import DatasetTypes
from ..data_manifest import DatasetManifest, ImageDataManifest, ImageLabelWithCategoryManifest
from ..data_reader import FileReader, PILImageLoader
from ..dataset.base_dataset import BaseDatasetInfo
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class VisionDataset(BaseDataset):
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

        assert dataset_manifest is not None
        assert coordinates in ['relative', 'absolute']

        super().__init__(dataset_info)

        self.dataset_manifest = dataset_manifest
        self.coordinates = coordinates
        self._file_reader = FileReader()
        self.dataset_resources = dataset_resources

    @property
    def labels(self):
        return self.dataset_manifest.categories

    def __len__(self):
        return len(self.dataset_manifest.images)

    def _get_single_item(self, index):
        image_manifest: ImageDataManifest = self.dataset_manifest.images[index]
        image = self._load_image(image_manifest.img_path)
        target = image_manifest.labels
        if self.coordinates == 'relative':
            w, h = image.size
            target = VisionDataset._convert_box_to_relative(image_manifest.labels, w, h, self.dataset_info)

        return image, target, str(index)

    def close(self):
        self._file_reader.close()

    def _load_image(self, filepath):
        full_path = filepath.replace('\\', '/')
        try:
            with self._file_reader.open(full_path, 'rb') as f:
                img = PILImageLoader.load_from_stream(f)
                logger.debug(f'Loaded image from path: {full_path}')
                return img
        except Exception:
            logger.exception(f'Failed to load an image with path: {full_path}')
            raise

    @staticmethod
    def _convert_box_to_relative(target: typing.Union[typing.List[ImageLabelWithCategoryManifest], dict], img_w, img_h, dataset_info):
        # Convert absolute coordinates to relative coordinates.
        # Example: for image with size (200, 200), (1, 100, 100, 200, 200) => (1, 0.5, 0.5, 1.0, 1.0)
        if dataset_info.type == DatasetTypes.MULTITASK:
            return {task_name: VisionDataset._convert_box_to_relative(task_target, img_w, img_h, dataset_info.sub_task_infos[task_name]) for task_name, task_target in target.items()}

        if dataset_info.type == DatasetTypes.IMAGE_OBJECT_DETECTION:
            relative_target = copy.deepcopy(target)
            for t in relative_target:
                l = t.label_data
                t.label_data = [l[0], l[1] / img_w, l[2] / img_h, l[3] / img_w, l[4] / img_h]
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

        assert dataset is not None
        assert local_cache_params
        assert local_cache_params.get('dir')
        local_cache_params['n_copies'] = local_cache_params.get('n_copies', 1)
        assert local_cache_params['n_copies'] >= 1, 'n_copies must be equal or greater than 1.'

        super().__init__(dataset.dataset_info)

        self._dataset = dataset
        self._local_cache_params = local_cache_params
        if not os.path.exists(self._local_cache_params['dir']):
            os.makedirs(self._local_cache_params['dir'])

        self._annotations = {}
        self._paths = {}

    @property
    def labels(self):
        return self._dataset.labels

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
        return pathlib.Path(self._local_cache_params['dir']) / f'{img_idx}.{img_format}'

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

        return DatasetManifest(images, self.labels, self._dataset.dataset_info.type)

    def close(self):
        self._dataset.close()
