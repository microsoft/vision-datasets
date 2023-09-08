import abc
import copy
import logging

from ....common.utils import deep_merge
from ..data_manifest import CategoryManifest, DatasetManifest, ImageLabelWithCategoryManifest
from .operation import Operation

logger = logging.getLogger(__name__)


class MergeStrategy(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def merge(self, *args: DatasetManifest):
        pass

    def check(self, *args: DatasetManifest):
        if len(args) < 1:
            raise ValueError('less than one manifest provided.')
        if any([arg is None for arg in args]):
            raise ValueError('"None" manifest found')


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


class SingleTaskMerge(MergeStrategy):
    """
    Merge for single task data type
    """

    def merge(self, *args: DatasetManifest):
        data_type = args[0].data_type
        images = []

        categories, category_name_to_idx = self._combine_categories(args) if bool(args[0].categories) else (None, None)

        for manifest in args:
            for image in manifest.images:
                new_image = copy.deepcopy(image)
                new_image.id = len(images)
                if categories:
                    for label in new_image.labels:
                        label: ImageLabelWithCategoryManifest = label
                        label.category_id = category_name_to_idx[manifest.categories[label.category_id].name]
                images.append(new_image)

        additional_info = deep_merge([x.additional_info for x in args])
        return DatasetManifest(images, categories, copy.deepcopy(data_type), additional_info)

    def check(self, *args: DatasetManifest):
        super().check(*args)

        if any([x.is_multitask for x in args]):
            raise ValueError('All manifests must be of the same data type and single task.')
        if any([x.data_type != args[0].data_type for x in args]):
            raise ValueError('All manifests must be of the same data type.')

    def _combine_categories(self, manifests: DatasetManifest):
        category_name_to_idx = {}
        for manifest in manifests:
            for category in manifest.categories:
                if category.name not in category_name_to_idx:
                    category_name_to_idx[category.name] = len(category_name_to_idx)
        categories = [CategoryManifest(i, x) for i, x in enumerate(category_name_to_idx.keys())]

        return categories, category_name_to_idx
