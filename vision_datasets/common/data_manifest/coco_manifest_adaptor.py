import json
import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Union

from ..data_reader import FileReader
from ..utils import can_be_url, construct_full_url_or_path_func
from .data_manifest import CategoryManifest, DatasetManifest, ImageDataManifest

logger = logging.getLogger(__name__)


class CocoManifestAdaptorBase(ABC):
    """
    Adaptor for generating manifest from coco format

    image paths should be stored under 'file_name'
    """

    def __init__(self, data_type: Union[dict, str]) -> None:
        super().__init__()
        self.data_type = data_type
        self._url_or_root_dir = None

    def create_dataset_manifest(self, coco_file_path_or_url: Union[str, dict, pathlib.Path], url_or_root_dir: str = None):
        """ construct a dataset manifest out of coco file
        Args:
            coco_file_path_or_url (str or pathlib.Path or dict): path or url to coco file. dict if multitask
            url_or_root_dir (str): container url or sas if resources are store in blob container, or a local dir
        """

        if not coco_file_path_or_url:
            return None

        self._url_or_root_dir = url_or_root_dir

        get_full_url_or_path = construct_full_url_or_path_func(self._url_or_root_dir)
        file_reader = FileReader()
        coco_file_path_or_url = coco_file_path_or_url if can_be_url(coco_file_path_or_url) else get_full_url_or_path(coco_file_path_or_url)
        with file_reader.open(coco_file_path_or_url, encoding='utf-8') as file_in:
            coco_manifest = json.load(file_in)
        file_reader.close()

        images_by_id = {img['id']: ImageDataManifest(img['id'], self._append_zip_prefix_if_needed(img, img['file_name']), img.get('width'),
                                                     img.get('height'), [], self._get_additional_info(img, {'id', 'file_name', 'width', 'height', 'zip_file'})) for img in coco_manifest['images']}

        images, categories = self.get_images_and_categories(images_by_id, coco_manifest)
        return DatasetManifest(images, categories, self.data_type, self._get_additional_info(coco_manifest, {'images', 'categories', 'annotations'}))

    @abstractmethod
    def get_images_and_categories(self, images_by_id, coco_manifest):
        pass

    def _append_zip_prefix_if_needed(self, info_dict: dict, file_name):
        get_full_url_or_path = construct_full_url_or_path_func(self._url_or_root_dir)
        zip_prefix = info_dict.get('zip_file', '')
        if zip_prefix:
            zip_prefix += '@'
            if can_be_url(self._url_or_root_dir):
                raise ValueError('Cannot read files in zip from blob directly. Please download the zip file to local folder first.')

        return get_full_url_or_path(zip_prefix + file_name)

    def _get_additional_info(self, data, to_exclude):
        return {x: data[x] for x in data if x not in to_exclude}


class CocoManifestWithCategoriesAdaptor(CocoManifestAdaptorBase):
    """
    Adaptor for generating manifest from coco format

    image paths should be stored under 'file_name'
    """

    def get_images_and_categories(self, images_by_id, coco_manifest):
        label_id_to_pos, categories = self._process_categories(coco_manifest['categories'])

        for annotation in coco_manifest['annotations']:
            img = images_by_id[annotation['image_id']]
            self.process_label(img, annotation, coco_manifest, label_id_to_pos)

        images = [x for x in images_by_id.values()]
        images.sort(key=lambda x: x.id)

        return images, categories

    @abstractmethod
    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict, label_id_to_pos):
        pass

    def _process_categories(self, coco_categories):
        cate_id_name = [(cate['id'], cate['name'], cate.get('supercategory'), self._get_additional_info(cate, {'id', 'name', 'supercategory'})) for cate in coco_categories]
        cate_id_name.sort(key=lambda x: x[0])

        label_id_to_pos = {x[0]: i for i, x in enumerate(cate_id_name)}
        categories = [CategoryManifest(i, x[1], x[2], x[3]) for i, x in enumerate(cate_id_name)]

        return label_id_to_pos, categories


class CocoManifestWithoutCategoriesAdaptor(CocoManifestAdaptorBase):
    """
    Adaptor for generating manifest from coco format

    image paths should be stored under 'file_name'
    """

    def get_images_and_categories(self, images_by_id, coco_manifest):
        for annotation in coco_manifest['annotations']:
            img = images_by_id[annotation['image_id']]
            self.process_label(img, annotation, coco_manifest)

        images = [x for x in images_by_id.values()]
        images.sort(key=lambda x: x.id)

        return images, None

    @abstractmethod
    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        pass
