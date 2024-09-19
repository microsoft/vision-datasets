import logging
from copy import deepcopy
from typing import Any, Dict, List

from vision_datasets.common import DatasetTypes, KeyValuePairDatasetInfo, VisionDataset
from vision_datasets.key_value_pair import (
    KeyValuePairDatasetManifest,
    KeyValuePairLabelManifest,
)

logger = logging.getLogger(__name__)


class DetectionAsKeyValuePairDatasetBase(VisionDataset):
    """Dataset class that access Detection datset as KeyValuePair dataset."""

    def __init__(self, detection_dataset: VisionDataset):
        """
        Initializes an instance of the DetectionAsKeyValuePairDataset class.
        Args:
            detection_dataset (VisionDataset): The detection dataset to convert to key-value pair dataset.
        """

        if detection_dataset is None or detection_dataset.dataset_info.type not in {DatasetTypes.IMAGE_OBJECT_DETECTION}:
            raise ValueError("DetectionAsKeyValuePairDataset only supports Image Object Detection datasets.")

        # Generate schema and update dataset info
        detection_dataset = deepcopy(detection_dataset)

        dataset_info_dict = detection_dataset.dataset_info.__dict__
        dataset_info_dict["type"] = DatasetTypes.KEY_VALUE_PAIR.name.lower()
        self.class_names = [c.name for c in detection_dataset.dataset_manifest.categories]
        self.class_id_to_names = {c.id: c.name for c in detection_dataset.dataset_manifest.categories}
        self.img_id_to_pos = {x.id: i for i, x in enumerate(detection_dataset.dataset_manifest.images)}

        schema = self.construct_schema(self.class_names)
        # Update dataset_info with schema
        dataset_info = KeyValuePairDatasetInfo({**dataset_info_dict, "schema": schema})

        # Construct KeyValuePairDatasetManifest
        annotations = []
        for id, img in enumerate(detection_dataset.dataset_manifest.images, 1):
            bboxes = [box.label_data for box in img.labels]

            kvp_label_data = self.construct_kvp_label_data(bboxes)
            img_ids = [self.img_id_to_pos[img.id]]  # 0-based index
            kvp_annotation = KeyValuePairLabelManifest(id, img_ids, label_data=kvp_label_data)

            # KVPDatasetManifest expects img.labels to be empty. Labels are instead stored in KVP annotation
            img.labels = []
            annotations.append(kvp_annotation)

        dataset_manifest = KeyValuePairDatasetManifest(detection_dataset.dataset_manifest.images, annotations, schema, additional_info=detection_dataset.dataset_manifest.additional_info)
        super().__init__(dataset_info, dataset_manifest, dataset_resources=detection_dataset.dataset_resources)

    def construct_schema(self, class_names: List[str]) -> Dict[str, Any]:
        schema: Dict[str, Any] = self.DETECTION_SCHEMA  # initialize with base schema
        schema["fieldSchema"][f"{self.OBJECTS_KEY}"]["items"]["classes"] = {c: {"description": f"Always output {c} as the class."} if len(class_names) == 1 else {} for c in class_names}
        return schema

    def construct_kvp_label_data(self, bboxes: List[List[int]]):
        raise NotImplementedError
    
    def sort_bboxes_label_wise(self, bboxes: List[List[int]]) -> Dict[str, List[List[int]]]:
        """
        Convert a list of bounding boxes to a dictionary with class name as key and list of bounding boxes as value.

        E.g. [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]] -> {"<label_name_1>": [[2, 3, 4, 5], [2, 3, 4, 5]], "<label_name_2>": [[3, 4, 5, 6]]}
        """

        label_wise_bboxes = {}
        for box in bboxes:
            label_name = self.class_id_to_names[box[0]]
            if label_name not in label_wise_bboxes:
                label_wise_bboxes[label_name] = []
            label_wise_bboxes[label_name].append(box[1:])

        return label_wise_bboxes


class DetectionAsKeyValuePairDataset(DetectionAsKeyValuePairDatasetBase):
    OBJECTS_KEY = "detectedObjects"
    DETECTION_SCHEMA = {
        "name": "Image Object Detection",
        "description": "Detect objects in images and provide bounding boxes and class label for each object",
        "fieldSchema": {
            f"{OBJECTS_KEY}": {
                "type": "array",
                "description": "Objects in the image of the specified classes, with bounding boxes",
                "items": {
                    "type": "string",
                    "description": "Class name of the object",
                    "classes": {},
                    "includeGrounding": True
                }
            }
        }
    }

    def construct_kvp_label_data(self, bboxes: List[List[int]]):
        """
        Convert the detection dataset label_name to the desired format for KVP annnotation as defined by the DETECTION_SCHEMA.
        E.g. {"fields": {"bboxes": {"value": [{"value": "class1", "groundings" : [[10,10,20,20]]},
                                              {"value": "class2", "groundings" : [[0,0,20,20], [20,20,30,30]]}]
            }
        """

        label_wise_bboxes = self.sort_bboxes_label_wise(bboxes)

        return {
            f"{KeyValuePairLabelManifest.LABEL_KEY}": {
                f"{self.OBJECTS_KEY}": {
                    f"{KeyValuePairLabelManifest.LABEL_VALUE_KEY}": [{f"{KeyValuePairLabelManifest.LABEL_VALUE_KEY}": key, f"{KeyValuePairLabelManifest.LABEL_GROUNDINGS_KEY}": value} for key, value in label_wise_bboxes.items()]
                }
            }
        }


class DetectionAsKeyValuePairDatasetForMultilabelClassification(DetectionAsKeyValuePairDatasetBase):
    OBJECTS_KEY = "objectClassNames"
    DETECTION_SCHEMA = {
        "name": "Get object class names from an image.",
        "description": "Find all the objects within the provided classes from an image. If multiple objects of the same class are present, only output the class name once.",
        "fieldSchema": {
            f"{OBJECTS_KEY}": {
                "type": "array",
                "description": "Unique class names of objects in the image of the specified classes.",
                "items": {
                    "type": "string",
                    "description": "Class name of the object.",
                    "classes": {}
                }
            }
        }
    }
    
    def construct_kvp_label_data(self, bboxes: List[List[int]]):
        """
        Convert the detection dataset label_name to the desired format for KVP annnotation as defined by the DETECTION_SCHEMA.
        E.g. {"fields": {"objectClassNames": {"value": [{"value": "class1"}, {"value": "class2"}]}}
        """
        
        label_wise_bboxes = self.sort_bboxes_label_wise(bboxes)

        return {
            f"{KeyValuePairLabelManifest.LABEL_KEY}": {
                f"{self.OBJECTS_KEY}": {
                    f"{KeyValuePairLabelManifest.LABEL_VALUE_KEY}": [{f"{KeyValuePairLabelManifest.LABEL_VALUE_KEY}": key} for key in label_wise_bboxes.keys()]
                }
            }
        }
