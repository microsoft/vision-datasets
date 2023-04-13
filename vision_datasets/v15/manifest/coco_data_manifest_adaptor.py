from abc import ABC, abstractmethod
import json
import logging
import pathlib
from typing import Union

from ...common.util import is_url, FileReader
from ...v15.common import BBoxFormat, DatasetTypes
from .data_manifest import DatasetManifest, ImageDataManifest
from .utils import generate_multitask_dataset_manifest, construct_full_url_or_path_generator

logger = logging.getLogger(__name__)


class CocoManifestAdaptorBase(ABC):
    """
    Adaptor for generating manifest from coco format

    image paths should be stored under 'file_name'
    """

    def __init__(self, data_type: Union[dict, str]) -> None:
        super().__init__()
        self.data_type = data_type

    def create_dataset_manifest(self, coco_file_path_or_url: Union[str, dict, pathlib.Path], container_sas_or_root_dir: str = None):
        """ construct a dataset manifest out of coco file
        Args:
            coco_file_path_or_url (str or pathlib.Path or dict): path or url to coco file. dict if multitask
            container_sas_or_root_dir (str): container sas if resources are store in blob container, or a local dir
        """

        if not coco_file_path_or_url:
            return None

        get_full_sas_or_path = construct_full_url_or_path_generator(container_sas_or_root_dir)
        file_reader = FileReader()
        coco_file_path_or_url = coco_file_path_or_url if is_url(coco_file_path_or_url) else get_full_sas_or_path(coco_file_path_or_url)
        with file_reader.open(coco_file_path_or_url, encoding='utf-8') as file_in:
            coco_manifest = json.load(file_in)

        file_reader.close()

        def append_zip_prefix_if_needed(info_dict: dict, file_name):
            zip_prefix = info_dict.get('zip_file', '')
            if zip_prefix:
                zip_prefix += '@'

            return get_full_sas_or_path(zip_prefix + file_name)

        self.append_zip_prefix_if_needed = append_zip_prefix_if_needed
        images_by_id = {img['id']: ImageDataManifest(img['id'], append_zip_prefix_if_needed(img, img['file_name']), img.get('width'), img.get('height'), [], {}) for img in coco_manifest['images']}

        images, labelmap = self.get_images_and_labelmap(images_by_id, coco_manifest)
        return DatasetManifest(images, labelmap, self.data_type)

    @abstractmethod
    def get_images_and_labelmap(self, images_by_id, coco_manifest):
        pass


class ManifestAdaptorFactory:
    _mapping = {}

    @classmethod
    def register(cls, data_type: str):
        """Class decorator to register a TaskProcessor to this factory."""
        def decorator(klass):
            cls._mapping[data_type] = klass
            return klass
        return decorator

    @classmethod
    def create(cls, data_type: str, *args, **kwargs) -> CocoManifestAdaptorBase:
        """Create TaskProcessor instance.

        The first argument must be a TaskConfig instance. The corresponding TaskProcessor will be created based on its class type.

        Args:
            task_config: TaskConfig
            *args, **kwargs
        """
        return cls._mapping[data_type](*args, **kwargs)


class CocoManifestWithCategoriesAdaptor(CocoManifestAdaptorBase):
    """
    Adaptor for generating manifest from coco format

    image paths should be stored under 'file_name'
    """

    def get_images_and_labelmap(self, images_by_id, coco_manifest):
        label_id_to_pos, labelmap = CocoManifestWithCategoriesAdaptor._process_categories(coco_manifest['categories'])

        for annotation in coco_manifest['annotations']:
            img = images_by_id[annotation['image_id']]
            self.process_label(img, annotation, coco_manifest, label_id_to_pos)

        images = [x for x in images_by_id.values()]
        images.sort(key=lambda x: x.id)

        return images, labelmap

    @abstractmethod
    def process_label(self, image, annotation, coco_manifest, label_id_to_pos):
        pass

    @staticmethod
    def _process_categories(coco_categories):
        cate_id_name = [(cate['id'], cate['name']) for cate in coco_categories]
        cate_id_name.sort(key=lambda x: x[0])

        label_id_to_pos = {x[0]: i for i, x in enumerate(cate_id_name)}
        labelmap = [x[1] for x in cate_id_name]

        return label_id_to_pos, labelmap


class CocoManifestWithoutCategoriesAdaptor(CocoManifestAdaptorBase):
    """
    Adaptor for generating manifest from coco format

    image paths should be stored under 'file_name'
    """

    def get_images_and_labelmap(self, images_by_id, coco_manifest):
        for annotation in coco_manifest['annotations']:
            img = images_by_id[annotation['image_id']]
            self.process_label(img, annotation, coco_manifest)

        images = [x for x in images_by_id.values()]
        images.sort(key=lambda x: x.id)

        return images, None

    @abstractmethod
    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        pass


@ManifestAdaptorFactory.register(DatasetTypes.MULTITASK)
class MultiTaskCocoManifestAdaptor(CocoManifestAdaptorBase):
    def __init__(self, data_types: dict) -> None:
        super().__init__(data_types)

    def create_dataset_manifest(self, coco_file_path_or_url: Union[str, dict, pathlib.Path], container_sas_or_root_dir: str = None):
        """ construct a dataset manifest out of coco file
        Args:
            coco_file_path_or_url (str or pathlib.Path or dict): path or url to coco file. dict if multitask
            container_sas_or_root_dir (str): container sas if resources are store in blob container, or a local dir
        """

        if not coco_file_path_or_url:
            return None

        assert isinstance(coco_file_path_or_url, dict)
        assert isinstance(self.data_type, dict)
        dataset_manifest_by_task = {k: ManifestAdaptorFactory.create(self.data_type[k]).create_dataset_manifest(coco_file_path_or_url[k], container_sas_or_root_dir)
                                    for k in coco_file_path_or_url}

        return generate_multitask_dataset_manifest(dataset_manifest_by_task)

    def get_images_and_labelmap(self, images_by_id, coco_manifest):
        pass


@ManifestAdaptorFactory.register(DatasetTypes.IC_MULTICLASS)
class MultiClassClassificationCocoManifestAdaptor(CocoManifestWithCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IC_MULTICLASS)

    def process_label(self, image: ImageDataManifest, annotation, coco_manifest, label_id_to_pos):
        assert len(image.labels) == 0, f"There should be exactly one label per image for {DatasetTypes.IC_MULTICLASS} datasets, but image with id {annotation['image_id']} has more than one."
        label = label_id_to_pos[annotation['category_id']]
        image.labels.append(label)


@ManifestAdaptorFactory.register(DatasetTypes.IC_MULTILABEL)
class MultiLabelClassificationCocoManifestAdaptor(CocoManifestWithCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IC_MULTILABEL)

    def process_label(self, image: ImageDataManifest, annotation, coco_manifest, label_id_to_pos):
        label = label_id_to_pos[annotation['category_id']]
        image.labels.append(label)


@ManifestAdaptorFactory.register(DatasetTypes.OD)
class ObjectDetectionCocoManifestAdaptor(CocoManifestWithCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IC_MULTICLASS)

    def process_label(self, image, annotation, coco_manifest, label_id_to_pos):
        bbox_format = coco_manifest.get('bbox_format')
        bbox_format = BBoxFormat[bbox_format] if bbox_format else BBoxFormat.LTWH

        c_id = label_id_to_pos[annotation['category_id']]
        bbox = annotation['bbox']
        bbox = bbox if bbox_format == BBoxFormat.LTRB else [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        label = [c_id] + bbox
        is_crowd = annotation.get('iscrowd', 0)
        image.labels.append(label)
        image.labels_extra_info['iscrowd'] = image.labels_extra_info.get('iscrowd', []) + [is_crowd]


@ManifestAdaptorFactory.register(DatasetTypes.IMCAP)
class ImageCaptionCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMCAP)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        image.labels.append(annotation['caption'])


@ManifestAdaptorFactory.register(DatasetTypes.IMAGE_TEXT_MATCHING)
class ImageTextMatchingCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_TEXT_MATCHING)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        image.labels.append((annotation['text'], annotation['match']))


@ManifestAdaptorFactory.register(DatasetTypes.IMAGE_MATTING)
class ImageMattingCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_MATTING)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        image.label_file_paths = image.label_file_paths or []
        image.label_file_paths.append(self.append_zip_prefix_if_needed(annotation, annotation['label']))


@ManifestAdaptorFactory.register(DatasetTypes.IMAGE_REGRESSION)
class ImageRegressionCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_REGRESSION)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        assert len(image.labels) == 0, f"There should be exactly one label per image for {DatasetTypes.IMAGE_REGRESSION} datasets, but image with id {annotation['image_id']} has more than one."
        image.labels.append(annotation['target'])


@ManifestAdaptorFactory.register(DatasetTypes.IMAGE_RETRIEVAL)
class ImageRetrievalCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_RETRIEVAL)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        image.labels.append(annotation['query'])
