import abc
import copy
import logging
from enum import Enum

from ..data_manifest import DatasetManifest, ImageLabelWithCategoryManifest, CategoryManifest
from .operation import Operation

logger = logging.getLogger(__name__)



class MergeStrategy(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def merge(self, *args: DatasetManifest):
        pass

    def check(self, *args: DatasetManifest):
        assert len(args) >= 1, 'less than one manifest provided.'
        assert all([arg is not None for arg in args]), '"None" manifest found'


class MergeStrategyType(Enum):
    IndependentImages = 0  # assuming images are independent, categories are uniquely determined by their names, if category exists
    IndependentAnnotations = 1  # assumsing annotations are independent, images are uniquely determined by their ids. A multitask dataset will be generated


class ManifestMerger(Operation):
    def __init__(self, strategy: MergeStrategy) -> None:
        super().__init__()
        self._strategy = strategy

    def run(self, *args: DatasetManifest):
        """
        Merge multiple data manifests of the same data type into one, with the assumptions that the images from different manifests are independent

        Args:
            args: manifests to be merged
        """
        self._strategy.check(*args)
        return self._strategy.merge(*args)


class SingleTaskMergeWithIndepedentImages(MergeStrategy):
    def merge(self, *args: DatasetManifest):
        data_type = args[0].data_type
        category_name_to_idx = {}
        images = []

        with_categories = bool(args[0].categories)
        if with_categories:
            for manifest in args:
                for category in manifest.categories:
                    if category.name not in category_name_to_idx:
                        category_name_to_idx[category.name] = len(category_name_to_idx)
            categories = [CategoryManifest(i, x) for i, x in enumerate(category_name_to_idx.keys())]
        else:
            categories = None    

        for manifest in args:
            for image in manifest.images:
                new_image = copy.deepcopy(image)
                new_image.id = len(images)
                if with_categories:
                    for label in new_image.labels:
                        label: ImageLabelWithCategoryManifest = label
                        label.category_id = category_name_to_idx[manifest.categories[label.category_id].name]
                images.append(new_image)

        return DatasetManifest(images, categories, copy.deepcopy(data_type))

    def check(self, *args: DatasetManifest):
        """Checking all category names are unique

        Raises:
            ValueError: if duplicate category name exists
        """
        super().check(*args)

        assert all([not x.is_multitask for x in args]), 'All manifests must be of the same data type and single task.'
        assert all([x.data_type == args[0].data_type for x in args]), 'All manifests must be of the same data type.'
