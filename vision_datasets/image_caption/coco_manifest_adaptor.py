from ..common import DatasetTypes, ImageDataManifest, CocoManifestAdaptorFactory, CocoManifestWithoutCategoriesAdaptor
from .manifest import ImageCaptionLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.IMAGE_CAPTION)
class ImageCaptionCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_CAPTION)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        image.labels.append(ImageCaptionLabelManifest(annotation['caption'], additional_info=self._get_additional_info(annotation, {'id', 'image_id', 'caption'})))
