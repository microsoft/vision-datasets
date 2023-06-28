from ..common import DatasetTypes
from ..common.data_manifest import ImageDataManifest
from ..common.data_manifest.coco_manifest_adaptor import CocoManifestWithCategoriesAdaptor
from ..common.factory import CocoManifestAdaptorFactory
from .manifest import ImageClassificationLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)
class MultiClassClassificationCocoManifestAdaptor(CocoManifestWithCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)

    def process_label(self, image: ImageDataManifest, annotation, coco_manifest, label_id_to_pos):
        if len(image.labels) != 0:
            raise ValueError(f"image with id {annotation['image_id']} has unexpected number of annotations {len(image.labels)} for {DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS} dataset.")

        label = ImageClassificationLabelManifest(label_id_to_pos[annotation['category_id']], additional_info=self._get_additional_info(annotation, {'id', 'image_id', 'category_id'}))
        image.labels.append(label)


@CocoManifestAdaptorFactory.register(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)
class MultiLabelClassificationCocoManifestAdaptor(CocoManifestWithCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)

    def process_label(self, image: ImageDataManifest, annotation, coco_manifest, label_id_to_pos):
        label = ImageClassificationLabelManifest(label_id_to_pos[annotation['category_id']], additional_info=self._get_additional_info(annotation, {'id', 'image_id', 'category_id'}))
        image.labels.append(label)
