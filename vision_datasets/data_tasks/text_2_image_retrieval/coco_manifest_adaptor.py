from ...common import DatasetTypes
from ...data_manifest import ImageDataManifest
from ...data_manifest.coco_manifest_adaptor import CocoManifestWithoutCategoriesAdaptor
from ...factory.coco_manifest_adaptor_factory import CocoManifestAdaptorFactory
from .manifest import Text2ImageRetrievalLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.TEXT_2_IMAGE_RETRIEVAL)
class Text2ImageRetrievalCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.TEXT_2_IMAGE_RETRIEVAL)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        image.labels.append(Text2ImageRetrievalLabelManifest(annotation['query']))
