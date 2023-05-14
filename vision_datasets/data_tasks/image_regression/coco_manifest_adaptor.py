from ...common import DatasetTypes
from ...data_manifest import ImageDataManifest
from ...data_manifest.coco_manifest_adaptor import CocoManifestWithoutCategoriesAdaptor
from ...factory.coco_manifest_adaptor_factory import CocoManifestAdaptorFactory
from .manifest import ImageRegressionLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.IMAGE_REGRESSION)
class ImageRegressionCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_REGRESSION)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        assert len(image.labels) == 0, f"There should be exactly one label per image for {DatasetTypes.IMAGE_REGRESSION} datasets, but image with id {annotation['image_id']} has more than one."
        image.labels.append(ImageRegressionLabelManifest(annotation['target']))
