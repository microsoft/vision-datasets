import abc
import logging
import pathlib
from typing import Dict, List, Union

from ..constants import DatasetTypes

logger = logging.getLogger(__name__)


class ManifestBase(abc.ABC):
    def __init__(self, additional_info: Dict = {}):
        self.additional_info = additional_info

    def __eq__(self, other):
        if not isinstance(other, ManifestBase):
            return False

        return self.additional_info == other.additional_info


class ImageLabelManifest(ManifestBase):
    def __init__(self, label_data=None, label_path: pathlib.Path = None, additional_info: Dict = {}):
        super().__init__(additional_info)

        if not ((label_data is None) ^ (label_path is None)):
            raise ValueError('Must provide either label data or file path to label data, but not both of them.')

        if label_data is not None:
            self._check_label(label_data)

        self._label_data = label_data
        self.label_path = label_path

    @property
    def label_data(self):
        if self._label_data is None:  # lazy load
            if self.label_path:
                self._label_data = self._read_label_data()
                self._check_label(self._label_data)
            else:
                return None

        return self._label_data

    @label_data.setter
    def label_data(self, val):
        self._label_data = val

    @abc.abstractmethod
    def _read_label_data(self):
        pass

    def __eq__(self, other):
        if not super().__eq__(other):
            return False

        if not isinstance(other, ImageLabelManifest):
            return False

        if self.label_path or other.label_path:
            # this is a blunt check, without taking the case two files containing the same image data into consideration
            return self.label_path == other.label_path

        return self._label_data == self._label_data

    @abc.abstractmethod
    def _check_label(self, label_data):
        raise NotImplementedError


class ImageLabelWithCategoryManifest(ImageLabelManifest):
    @property
    @abc.abstractmethod
    def category_id(self):
        pass

    @category_id.setter
    @abc.abstractmethod
    def category_id(self, value):
        pass

    def _category_id_check(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError


class MultiImageLabelManifest(ImageLabelManifest, abc.ABC):
    """
    Encapsulates information of an multi-image label. The label can be associated with a single image, or multi-image jointly. 
    """
    def __init__(self,
                 id: int,
                 img_ids: List[int],
                 label_data,
                 additional_info: Dict = {}):
        """
        Annotation manifest organized by img_ids, each set of images can have multiple labels. 
        Args:
            id (int): annotation id
            img_ids (list of integers): image ids
            label_data (dict or str): label data
            additional_info (dict): additional info about this annotation
        """
        self.id = id
        self.img_ids = img_ids
        super().__init__(label_data, label_path=None, additional_info=additional_info)

    def __eq__(self, other) -> bool:
        if not super().__eq__(other):
            return False

        if not isinstance(other, MultiImageLabelManifest):
            return False

        return self.id == other.id and self.img_ids == other.img_ids and self.label_data == other.label_data \
            and self.additional_info == other.additional_info

    def is_negative(self) -> bool:
        return self.label_data is None


class ImageDataManifest(ManifestBase):
    """
    Encapsulates the information and annotations of an image.

    img_path could be 1. a local path 2. a local path in a non-compressed zip file (`c:\a.zip@1.jpg`) or 3. a url.
    """

    def __init__(self,
                 id: Union[str, int],
                 img_path: Union[pathlib.Path, str],
                 width: int,
                 height: int,
                 labels: Union[List[ImageLabelManifest], Dict[str, List[ImageLabelManifest]]],
                 additional_info: Dict = {}):
        """
        Args:
            id (int or str): image id
            img_path (str): path to image
            width (int): image width
            height (int): image height
            labels (list or dict): labels for the image
            additional_info (dict): additional info about this image
        """
        super().__init__(additional_info)

        self.id = id
        self.img_path = img_path
        self.width = width
        self.height = height
        self.labels = labels

    def __eq__(self, other) -> bool:
        if not super().__eq__(other):
            return False

        if not isinstance(other, ImageDataManifest):
            return False

        return self.id == other.id and self.img_path == other.img_path and self.width == other.width \
            and self.height == other.height and self.labels == other.labels

    def is_negative(self) -> bool:
        if not self.labels:
            return True

        if not isinstance(self.labels, dict):
            return False

        return not any(labels for labels in self.labels.values())


class CategoryManifest(ManifestBase):
    def __init__(self, id, name: str, super_category: str = None, addtional_info={}):
        super().__init__(addtional_info)

        self.id = id
        self.name = name
        self.super_category = super_category

    def __eq__(self, other) -> bool:
        if not super().__eq__(other):
            return False

        if not isinstance(other, CategoryManifest):
            return False

        return self.id == other.id and self.name == other.name and self.super_category == other.super_category


class DatasetManifest(ManifestBase):
    """
    Encapsulates information about a dataset including images, categories (if applicable), and annotations. Information about each image is encapsulated in ImageDataManifest.
    """

    def __init__(self, images: List[ImageDataManifest], categories: Union[List[CategoryManifest], Dict[str, List[CategoryManifest]]], data_type: Union[str, dict], addtional_info={}):
        """

        Args:
            images (list): image manifest
            categories (list or dict): labels or labels by task name
            data_type (str or dict) : data type, or data type by task name
            additional_info (dict): additional info about this dataset
        """
        if not data_type or data_type == DatasetTypes.MULTITASK:
            raise ValueError('For multitask, data_type should be a dict mapping task name to concrete data type.')

        if isinstance(categories, dict):
            if not isinstance(data_type, dict):
                raise ValueError('categories being a dict indicating this is a multitask dataset, however the data_type is not a dict.')
            if categories.keys() != data_type.keys():
                raise ValueError(f'mismatched task names in categories and task_type: {categories.keys()} vs {data_type.keys()}')

        super().__init__(addtional_info)

        self.images = images
        self.categories = categories
        self.data_type = data_type

    @property
    def is_multitask(self):
        """
        is this dataset multi-task dataset or not
        """

        return isinstance(self.data_type, dict)

    def __eq__(self, other) -> bool:
        if not isinstance(other, DatasetManifest):
            return False

        return self.images == other.images and self.categories == other.categories and self.data_type == other.data_type

    def __len__(self):
        return len(self.images)


class DatasetManifestWithMultiImageLabel(ManifestBase):
    """
    Multi-image dataset manifest supporting multi-image sample. Image information except label is encapsulated in ImageDataManifest. Annotation information is encapsulated in MultiImageLabelManifest, 
    including field 'img_ids' to capture indices in images list, and label_data to capture the label.
    """

    def __init__(self, images: List[ImageDataManifest], annotations: List[MultiImageLabelManifest], data_type: str, addtional_info={}):
        """

        Args:
            images (list of ImageDataManifest): image manifest. For each single image, image.labels won't be respected, label information is instead obtained in annotations below.
            annotations (list): annotations
            data_type (str) : data type
            additional_info (dict): additional info about this dataset
        """
        if isinstance(data_type, dict):
            raise ValueError("composition type is not supported!")
        for img in images:
            if img.labels:
                raise ValueError(f"labels associated with single image manifest ({img.id}) should not be provided.")
        for ann in annotations:
            if not ann.is_negative() and any([img_id < 0 or img_id >= len(images) for img_id in ann.img_ids]):
                raise ValueError(f"image ids of annotation are out of range, expect 0 to {len(images)-1}, got {ann.img_ids}!")
        super().__init__(addtional_info)

        self.images = images
        self.annotations = annotations
        self.data_type = data_type
        self.categories = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, DatasetManifestWithMultiImageLabel) or self.data_type != other.data_type:
            return False
        return self.images == other.images and self.annotations == other.annotations

    def __len__(self):
        return len(self.annotations)
