from copy import deepcopy
import logging
from typing import Any, Dict

from vision_datasets.common import DatasetTypes, KeyValuePairDatasetInfo, VisionDataset
from vision_datasets.key_value_pair import (
    KeyValuePairDatasetManifest,
    KeyValuePairLabelManifest,
)

logger = logging.getLogger(__name__)


class VQAAsKeyValuePairDataset(VisionDataset):
    """Dataset class that access Visual Question Answering (VQA) datset as KeyValuePair dataset."""

    ANSWER_KEY = "answer"
    RATIONALE_KEY = "rationale"
    QUESTION_KEY = "question"

    def __init__(self, vqa_dataset: VisionDataset):
        """
        Initializes an instance of the VQAAsKeyValuePairDataset class.
        Args:
            vqa_dataset (VisionDataset): The VQA dataset to convert to key-value pair dataset.
        """

        if vqa_dataset is None or vqa_dataset.dataset_info.type is not DatasetTypes.VISUAL_QUESTION_ANSWERING:
            raise ValueError("Input dataset must be a Visual Question Answering dataset.")

        # Generate schema and update dataset info
        vqa_dataset = deepcopy(vqa_dataset)

        dataset_info_dict = deepcopy(vqa_dataset.dataset_info.__dict__)
        dataset_info_dict["type"] = DatasetTypes.KEY_VALUE_PAIR.name.lower()

        schema = self._schema
        # Update dataset_info with schema
        dataset_info = KeyValuePairDatasetInfo({**dataset_info_dict, "schema": schema})

        dataset_manifest = vqa_dataset.dataset_manifest
        self.img_id_to_pos = {x.id: i for i, x in enumerate(dataset_manifest.images)}

        # Construct KeyValuePairDatasetManifest
        annotations = []
        id = 1
        for _, img in enumerate(dataset_manifest.images, 1):
            label_data = [label.label_data for label in img.labels]

            for label in label_data:
                kvp_label_data = self.construct_kvp_label_data(label)
                img_ids = [self.img_id_to_pos[img.id]]  # 0-based index
                kvp_annotation = KeyValuePairLabelManifest(id, img_ids, label_data=kvp_label_data)
                id += 1

                # KVPDatasetManifest expects img.labels to be empty. Labels are instead stored in KVP annotation
                img.labels = []
                annotations.append(kvp_annotation)

        dataset_manifest = KeyValuePairDatasetManifest(deepcopy(dataset_manifest.images), annotations, schema, additional_info=deepcopy(vqa_dataset.dataset_manifest.additional_info))
        super().__init__(dataset_info, dataset_manifest, dataset_resources=vqa_dataset.dataset_resources)

    @property
    def _schema(self) -> Dict[str, Any]:
        return {
            "name": "Visual Question Answering",
            "description": "Answer questions on given images and provide rationale for the answer.",
            "fieldSchema": {
                self.ANSWER_KEY: {
                    "type": "string",
                    "description": "Answer to the question.",
                },
                self.RATIONALE_KEY: {
                    "type": "string",
                    "description": "Rationale for the answer.",
                },
            }
        }

    def construct_kvp_label_data(self, label: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """
        Convert the VQA dataset label to the desired format for KVP annotation as defined by the SCHEMA_BASE.
        E.g. {"fields":
                      {"answer": {"value": "yes"}},
              "text": {"question": "Is there a dog in the image?"}
            }
        """

        if self.QUESTION_KEY not in label:
            raise KeyError(f"Question key '{self.QUESTION_KEY}' not found in label.")
        if self.ANSWER_KEY not in label:
            raise KeyError(f"Answer key '{self.ANSWER_KEY}' not found in label.")

        kvp_label_data = {
            KeyValuePairLabelManifest.LABEL_KEY: {
                self.ANSWER_KEY: {KeyValuePairLabelManifest.LABEL_VALUE_KEY: label[self.ANSWER_KEY]},
            },
            KeyValuePairLabelManifest.TEXT_INPUT_KEY: {self.QUESTION_KEY: label[self.QUESTION_KEY]},
        }

        return kvp_label_data
