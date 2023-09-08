import abc
import copy

from ..data_manifest import DatasetManifest, ImageDataManifest
from .operation import Operation


class ImageFilter(abc.ABC):
    @abc.abstractmethod
    def should_be_filtered(self, image: ImageDataManifest, data_manifest: DatasetManifest) -> bool:
        pass


class DatasetFilter(Operation):
    """
    Filter images by certain conditions
    """
    def __init__(self, image_filter: ImageFilter):
        self.image_filter = image_filter

    def run(self, *args: DatasetManifest):
        if len(args) != 1:
            raise ValueError

        manifest = args[0]
        return DatasetManifest(copy.deepcopy([x for x in manifest.images if not self.image_filter.should_be_filtered(x, manifest)]),
                               copy.deepcopy(manifest.categories),
                               copy.deepcopy(manifest.data_type),
                               copy.deepcopy(manifest.additional_info))


class ImageNoAnnotationFilter(ImageFilter):
    def should_be_filtered(self, image: ImageDataManifest, data_manifest: DatasetManifest) -> bool:
        return image.is_negative()
