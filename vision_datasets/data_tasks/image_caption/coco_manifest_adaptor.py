from ...common import DatasetTypes
from ...data_manifest import ImageDataManifest
from ...data_manifest.coco_manifest_adaptor import CocoManifestWithoutCategoriesAdaptor
from ...factory.coco_manifest_adaptor_factory import CocoManifestAdaptorFactory
from .manifest import ImageCaptionLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.IMAGE_CAPTION)
class ImageCaptionCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_CAPTION)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        image.labels.append(ImageCaptionLabelManifest(annotation['caption']))
