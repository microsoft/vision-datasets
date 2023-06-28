from ..common import DatasetTypes, ImageDataManifest, CocoManifestWithoutCategoriesAdaptor, CocoManifestAdaptorFactory
from .manifest import ImageMattingLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.IMAGE_MATTING)
class ImageMattingCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_MATTING)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        image.labels.append(ImageMattingLabelManifest(label_path=self._append_zip_prefix_if_needed(annotation, annotation['label']),
                                                      additional_info=self._get_additional_info(annotation, {'id', 'image_id', 'label'})))
