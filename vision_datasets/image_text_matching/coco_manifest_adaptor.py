from ..common import DatasetTypes, ImageDataManifest, CocoManifestWithoutCategoriesAdaptor, CocoManifestAdaptorFactory
from .manifest import ImageTextMatchingLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.IMAGE_TEXT_MATCHING)
class ImageTextMatchingCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_TEXT_MATCHING)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        image.labels.append(ImageTextMatchingLabelManifest((annotation['text'], annotation['match']), additional_info=self._get_additional_info(annotation, {'id', 'image_id', 'text', 'match'})))
