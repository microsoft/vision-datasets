from ...common import DatasetTypes
from ...data_manifest import ImageDataManifest
from ...data_manifest.coco_manifest_adaptor import CocoManifestWithCategoriesAdaptor
from ...factory import CocoManifestAdaptorFactory
from .manifest import ImageClassificationLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)
class MultiClassClassificationCocoManifestAdaptor(CocoManifestWithCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)

    def process_label(self, image: ImageDataManifest, annotation, coco_manifest, label_id_to_pos):
        assert len(
            image.labels) == 0, f"There should be exactly one label per image for {DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS} datasets, "\
            f"but image with id {annotation['image_id']} has more than one."
        label = ImageClassificationLabelManifest(label_id_to_pos[annotation['category_id']])
        image.labels.append(label)


@CocoManifestAdaptorFactory.register(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)
class MultiLabelClassificationCocoManifestAdaptor(CocoManifestWithCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)

    def process_label(self, image: ImageDataManifest, annotation, coco_manifest, label_id_to_pos):
        label = ImageClassificationLabelManifest(label_id_to_pos[annotation['category_id']])
        image.labels.append(label)
