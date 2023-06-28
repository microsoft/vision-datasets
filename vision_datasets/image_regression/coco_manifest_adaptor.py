from ..common import DatasetTypes, ImageDataManifest, CocoManifestWithoutCategoriesAdaptor, CocoManifestAdaptorFactory
from .manifest import ImageRegressionLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.IMAGE_REGRESSION)
class ImageRegressionCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_REGRESSION)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        if len(image.labels) != 0:
            raise ValueError(f"image with id {annotation['image_id']} has unexpected number of annotations {len(image.labels)} for {DatasetTypes.IMAGE_REGRESSION} dataset.")
        image.labels.append(ImageRegressionLabelManifest(annotation['target'], additional_info=self._get_additional_info(annotation, {'id', 'image_id', 'target'})))
