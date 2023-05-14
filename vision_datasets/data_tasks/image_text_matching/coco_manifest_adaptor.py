from ...common import DatasetTypes
from ...data_manifest import ImageDataManifest
from ...data_manifest.coco_manifest_adaptor import CocoManifestWithoutCategoriesAdaptor
from ...factory import CocoManifestAdaptorFactory
from .manifest import ImageTextMatchingLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.IMAGE_TEXT_MATCHING)
class ImageTextMatchingCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_TEXT_MATCHING)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        image.labels.append(ImageTextMatchingLabelManifest((annotation['text'], annotation['match'])))
