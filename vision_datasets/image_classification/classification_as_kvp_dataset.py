import logging
import typing
from copy import deepcopy

from vision_datasets.common import DatasetTypes, KeyValuePairDatasetInfo, VisionDataset
from vision_datasets.key_value_pair import (
    KeyValuePairDatasetManifest,
    KeyValuePairLabelManifest,
)

logger = logging.getLogger(__name__)


CLASS_NAME_KEY = "className"
BASE_CLASSIFICATION_SCHEMA = {
    "name": "Multiclass image classification",
    "description": "Classify images into one of the provided classes.",
    "fieldSchema": {
            f"{CLASS_NAME_KEY}": {
                "type": "string",
                "description": "Class name that the image belongs to.",
                "classes": {}
            }
    }
}


class ClassificationAsKeyValuePairDataset(VisionDataset):
    """Dataset class that access Classification datset as KeyValuePair dataset."""

    def __init__(self, classification_dataset: VisionDataset):
        """
        Initializes an instance of the ClassificationAsKeyValuePairDataset class.
        Args:
            classification_dataset (VisionDataset): The classification dataset to convert to key-value pair dataset.
        """

        if classification_dataset is None or classification_dataset.dataset_info.type not in {DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS}:
            # TODO: Add support for multilabel classification
            raise ValueError

        # Generate schema and update dataset info
        classification_dataset = deepcopy(classification_dataset)

        dataset_info_dict = classification_dataset.dataset_info.__dict__
        dataset_info_dict["type"] = DatasetTypes.KEY_VALUE_PAIR.name.lower()
        self.class_names = [c.name for c in classification_dataset.dataset_manifest.categories]
        self.class_id_to_names = {c.id: c.name for c in classification_dataset.dataset_manifest.categories}
        self.img_id_to_pos = {x.id: i for i, x in enumerate(classification_dataset.dataset_manifest.images)}

        schema = self.construct_schema(self.class_names)
        # Update dataset_info with schema
        dataset_info = KeyValuePairDatasetInfo({**dataset_info_dict, "schema": schema})

        # Construct KeyValuePairDatasetManifest
        annotations = []
        for id, img in enumerate(classification_dataset.dataset_manifest.images, 1):
            label_id = img.labels[0].label_data
            label_name = self.class_id_to_names[label_id]

            kvp_label_data = self.construct_kvp_label_data(label_name)
            img_ids = [self.img_id_to_pos[img.id]]  # 0-based index
            kvp_annotation = KeyValuePairLabelManifest(id, img_ids, label_data=kvp_label_data)

            # KVPDatasetManifest expects img.labels to be empty. Labels are instead stored in KVP annotation
            img.labels = []
            annotations.append(kvp_annotation)

        dataset_manifest = KeyValuePairDatasetManifest(classification_dataset.dataset_manifest.images, annotations, schema, additional_info=classification_dataset.dataset_manifest.additional_info)
        super().__init__(dataset_info, dataset_manifest, dataset_resources=classification_dataset.dataset_resources)

    def construct_schema(self, class_names: typing.List[str]) -> typing.Dict[str, typing.Any]:
        schema: typing.Dict[str, typing.Any] = BASE_CLASSIFICATION_SCHEMA  # initialize with base schema
        schema["fieldSchema"][f"{CLASS_NAME_KEY}"]["classes"] = {c: {"description": f"A single class name. Only output {c} as the class name if present."} for c in class_names}
        return schema

    def construct_kvp_label_data(self, label_name: str) -> typing.Dict[str, typing.Union[typing.Dict[str, typing.Dict[str, str]], None]]:
        """
        Convert the classification dataset label_name to the desired format for KVP annnotation as defined by the BASE_CLASSIFICATION_SCHEMA.
        E.g. {"fields": {"className": {"value": <label_name>}}}

        """
        return {
            f"{KeyValuePairLabelManifest.LABEL_KEY}": {
                f"{CLASS_NAME_KEY}": {
                    f"{KeyValuePairLabelManifest.LABEL_VALUE_KEY}": label_name
                }
            }
        }
