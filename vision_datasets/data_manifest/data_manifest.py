import abc
import json
import logging
import pathlib
from typing import Dict, List, Union

from ..common import DatasetTypes

logger = logging.getLogger(__name__)


class ImageLabelManifest(abc.ABC):
    def __init__(self, label_data=None, label_path: pathlib.Path = None, additional_info: Dict = None):
        assert label_data is not None or label_path is not None, 'must provide either label data or file path to label data'

        self._label_data = label_data
        self.label_path = label_path
        self.additional_info = additional_info

    @property
    def label_data(self):
        if self._label_data is None:  # lazy load
            if self.label_path:
                self._label_data = self._read_label_data()
            else:
                return None

        return self._label_data

    @label_data.setter
    def label_data(self, val):
        self._label_data = val

    def _read_label_data(self):
        raise NotImplementedError

    def __setstate__(self, state):
        self._label_data = state['_label_data']
        self.label_path = state['label_path']
        self.additional_info = state['additional_info']

    def __getstate__(self):
        return {'_label_data': self.label_data, 'label_path': self.label_path, 'additional_info': self.additional_info}

    def __eq__(self, other):
        if not isinstance(other, ImageLabelManifest) or self.additional_info != other.additional_info:
            return False

        if self.label_path or other.label_path:
            # this is a blunt check, without taking the case two files containing the same image data into consideration
            return self.label_path == other.label_path

        return self._label_data == self._label_data

    def __str__(self) -> str:
        return f'Label: {json.dumps(self.__getstate__())}'


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
        assert isinstance(value, int) and value >= 0


class ImageDataManifest:
    """
    Encapsulates the information and annotations of an image.

    img_path could be 1. a local path 2. a local path in a non-compressed zip file (`c:\a.zip@1.jpg`) or 3. a url.
    """

    def __init__(self,
                 id: Union[str, int],
                 img_path: Union[pathlib.Path, str],
                 width: int,
                 height: int,
                 labels: Union[List[ImageLabelManifest], Dict[str, List[ImageLabelManifest]]]):
        """
        Args:
            id (int or str): image id
            img_path (str): path to image
            width (int): image width
            height (int): image height
            labels (list or dict): labels for the image
        """

        self.id = id
        self.img_path = img_path
        self.width = width
        self.height = height
        self.labels = labels

    def __eq__(self, other) -> bool:
        if not isinstance(other, ImageDataManifest):
            return False

        return self.id == other.id and self.img_path == other.img_path and self.width == other.width \
            and self.height == other.height and self.labels == other.labels

    def is_negative(self) -> bool:
        if not self.labels:
            return True

        if not isinstance(self.labels, dict):
            return False

        return sum([len(labels) for labels in self.labels.values()]) == 0


class CategoryManifest:
    def __init__(self, id, name: str, super_category: str = None):
        self.id = id
        self.name = name
        self.super_category = super_category

    def __eq__(self, other) -> bool:
        return self.id == other.id and self.name == other.name and self.super_category == other.super_category


class DatasetManifest:
    """
    Encapsulates information about a dataset including images, categories (if applicable), and annotations. Information about each image is encapsulated in ImageDataManifest.
    """

    def __init__(self, images: List[ImageDataManifest], categories: Union[List[CategoryManifest], Dict[str, List[CategoryManifest]]], data_type: Union[str, dict]):
        """

        Args:
            images (list): image manifest
            categories (list or dict): labels or labels by task name
            data_type (str or dict) : data type, or data type by task name
        """

        assert data_type and data_type != DatasetTypes.MULTITASK, 'For multitask, data_type should be a dict mapping task name to concrete data type.'

        if isinstance(categories, dict):
            assert isinstance(data_type, dict), 'categories being a dict indicating this is a multitask dataset, however the data_type is not a dict.'
            assert categories.keys() == data_type.keys(), f'mismatched task names in categories and task_type: {categories.keys()} vs {data_type.keys()}'

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
