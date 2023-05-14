from ...common import DatasetTypes
from ...data_manifest import DatasetTypes, ImageDataManifest
from ...data_manifest.coco_manifest_adaptor import CocoManifestWithoutCategoriesAdaptor
from ...factory import CocoManifestAdaptorFactory
from .manifest import ImageMattingLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.IMAGE_MATTING)
class ImageMattingCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_MATTING)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        image.labels.append(ImageMattingLabelManifest(label_path=self.append_zip_prefix_if_needed(annotation, annotation['label'])))
