import logging
import typing
from abc import ABC, abstractmethod
from copy import deepcopy

from vision_datasets.common import DatasetTypes, KeyValuePairDatasetInfo, VisionDataset
from vision_datasets.key_value_pair import (
    KeyValuePairDatasetManifest,
    KeyValuePairLabelManifest,
)

logger = logging.getLogger(__name__)


class ClassificationAsKeyValuePairDatasetBase(VisionDataset, ABC):
    """Dataset class that access Classification datset as KeyValuePair dataset."""

    def __init__(self, classification_dataset: VisionDataset, class_type_name: str):
        """
        Initializes an instance of the ClassificationAsKeyValuePairDataset class.
        Args:
            classification_dataset (VisionDataset): The classification dataset to convert to key-value pair dataset.
            class_type_name (str): The name of the class type in the schema, such as "AnimalKind", "DefectType", etc.
        """

        if classification_dataset is None or classification_dataset.dataset_info.type not in {DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL}:
            raise ValueError

        self.class_type_name = class_type_name
        # Generate schema and update dataset info
        classification_dataset = deepcopy(classification_dataset)

        dataset_info_dict = classification_dataset.dataset_info.__dict__
        dataset_info_dict["type"] = DatasetTypes.KEY_VALUE_PAIR.name.lower()
        self.class_names = [c.name for c in classification_dataset.dataset_manifest.categories]
        self.class_id_to_names = {c.id: c.name for c in classification_dataset.dataset_manifest.categories}
        self.img_id_to_pos = {x.id: i for i, x in enumerate(classification_dataset.dataset_manifest.images)}

        schema = self.create_schema_with_class_names(self.class_names)
        # Update dataset_info with schema
        dataset_info = KeyValuePairDatasetInfo({**dataset_info_dict, "schema": schema})

        # Construct KeyValuePairDatasetManifest
        annotations = []
        for id, img in enumerate(classification_dataset.dataset_manifest.images, 1):
            label_ids = [label.label_data for label in img.labels]
            label_names = [self.class_id_to_names[id] for id in label_ids]

            kvp_label_data = self.construct_kvp_label_data(label_names)
            img_ids = [self.img_id_to_pos[img.id]]  # 0-based index
            kvp_annotation = KeyValuePairLabelManifest(id, img_ids, label_data=kvp_label_data)

            # KVPDatasetManifest expects img.labels to be empty. Labels are instead stored in KVP annotation
            img.labels = []
            annotations.append(kvp_annotation)

        dataset_manifest = KeyValuePairDatasetManifest(classification_dataset.dataset_manifest.images, annotations, schema, additional_info=classification_dataset.dataset_manifest.additional_info)
        super().__init__(dataset_info, dataset_manifest, dataset_resources=classification_dataset.dataset_resources)

    @abstractmethod
    def create_schema_with_class_names(self, class_names: typing.List[str]) -> typing.Dict[str, typing.Any]:
        raise NotImplementedError

    @abstractmethod
    def construct_kvp_label_data(self, label_names: typing.List[str]) -> typing.Dict[str, typing.Union[typing.Dict[str, typing.Dict[str, str]], None]]:
        raise NotImplementedError


class MulticlassClassificationAsKeyValuePairDataset(ClassificationAsKeyValuePairDatasetBase):
    _DEFAULT_FIELD_NAME = "className"

    def __init__(self, classification_dataset, class_type_name=_DEFAULT_FIELD_NAME):
        super().__init__(classification_dataset, class_type_name)

    def create_schema_with_class_names(self, class_names: typing.List[str]) -> typing.Dict[str, typing.Any]:
        schema: typing.Dict[str, typing.Any] = self._schema
        schema["fieldSchema"][self.class_type_name]["classes"] = {c: {"description": f"A single class name. Only output {c} as the class name if present."} for c in class_names}
        return schema

    def construct_kvp_label_data(self, label_names: typing.List[str]) -> typing.Dict[str, typing.Union[typing.Dict[str, typing.Dict[str, str]], None]]:
        """
        Convert the classification dataset label_name to the desired format for KVP annotation as defined by the self.schema.
        E.g. {"fields": {"className": {"value": <label_name>}}}

        """
        return {
            KeyValuePairLabelManifest.LABEL_KEY: {
                self.class_type_name: {
                    KeyValuePairLabelManifest.LABEL_VALUE_KEY: label_names[0]
                }
            }
        }

    @property
    def _schema(self):
        return {
            "name": "Multiclass image classification",
            "description": "Classify images into one of the provided classes.",
            "fieldSchema": {
                self.class_type_name: {
                    "type": "string",
                    "description": "Class name that the image belongs to.",
                    "classes": {}
                }
            }
        }


class MultilabelClassificationAsKeyValuePairDataset(ClassificationAsKeyValuePairDatasetBase):
    _DEFAULT_FIELD_NAME = "classNames"

    def __init__(self, classification_dataset, class_type_name=_DEFAULT_FIELD_NAME):
        super().__init__(classification_dataset, class_type_name)

    def create_schema_with_class_names(self, class_names: typing.List[str]) -> typing.Dict[str, typing.Any]:
        schema: typing.Dict[str, typing.Any] = self.schema  # initialize with base schema
        schema["fieldSchema"][self.class_type_name]['items']["classes"] = {c: {"description": f"A single class name. Only output {c} as the class name if present."} for c in class_names}
        return schema

    def construct_kvp_label_data(self, label_names: typing.List[str]) -> typing.Dict[str, typing.Union[typing.Dict[str, typing.Dict[str, str]], None]]:
        """
        Convert the classification dataset label_name to the desired format for KVP annnotation as defined by the self.schema.
        E.g. {"fields": {"className": {"value": <label_name>}}}

        """
        return {
            KeyValuePairLabelManifest.LABEL_KEY: {
                self.class_type_name: {
                    KeyValuePairLabelManifest.LABEL_VALUE_KEY: [{KeyValuePairLabelManifest.LABEL_VALUE_KEY: n} for n in label_names]
                }
            }
        }

    @property
    def schema(self):
        return {
            "name": "Multilabel image classification",
            "description": "Classify images into one or more of the provided classes.",
            "fieldSchema": {
                self.class_type_name: {
                    "type": "array",
                    "description": "Class names that the image belongs to.",
                    "items": {
                        "type": "string",
                        'description': 'Single class name.',
                        "classes": {}
                    }
                }
            }
        }
