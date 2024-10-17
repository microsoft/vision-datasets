import unittest

from tests.test_fixtures import DetectionTestFixtures
from vision_datasets.common.constants import DatasetTypes
from vision_datasets.image_object_detection import (
    DetectionAsKeyValuePairDataset,
    DetectionAsKeyValuePairDatasetForMultilabelClassification,
)
from vision_datasets.key_value_pair.manifest import KeyValuePairLabelManifest


class TestDetectionAsKeyValuePairDataset(unittest.TestCase):
    def test_detection_to_kvp(self):
        sample_detection_dataset, tempdir = DetectionTestFixtures.create_an_od_dataset()
        with tempdir:
            kvp_dataset = DetectionAsKeyValuePairDataset(sample_detection_dataset)

            self.assertIsInstance(kvp_dataset, DetectionAsKeyValuePairDataset)
            self.assertEqual(kvp_dataset.dataset_info.type, DatasetTypes.KEY_VALUE_PAIR)
            self.assertIn("name", kvp_dataset.dataset_info.schema)
            self.assertIn("description", kvp_dataset.dataset_info.schema)
            self.assertIn("fieldSchema", kvp_dataset.dataset_info.schema)

            self.assertEqual(
                kvp_dataset.dataset_info.schema["fieldSchema"],
                {
                    "detectedObjects": {
                        "type": "array",
                        "description": "Objects in the image of the specified classes, with bounding boxes",
                        "items": {
                            "type": "string",
                            "description": "Class name of the object",
                            "classes": {
                                "1-class": {},
                                "2-class": {},
                                "3-class": {},
                                "4-class": {},
                            },
                            "includeGrounding": True,
                        },
                    }
                },
            )

            _, target, _ = kvp_dataset[0]
            self.assertIsInstance(target, KeyValuePairLabelManifest)
            self.assertEqual(
                target.label_data,
                {
                    "fields": {
                        "detectedObjects": {
                            "value": [
                                {"value": "1-class", "groundings": [[0, 0, 100, 100]]},
                                {"value": "2-class", "groundings": [[10, 10, 50, 100]]},
                            ]
                        }
                    }
                },
            )

    def test_single_class_description(self):
        sample_detection_dataset, tempdir = DetectionTestFixtures.create_an_od_dataset(
            n_categories=1
        )
        with tempdir:
            kvp_dataset = DetectionAsKeyValuePairDataset(sample_detection_dataset)

            self.assertEqual(
                kvp_dataset.dataset_info.schema["fieldSchema"]["detectedObjects"][
                    "items"
                ]["classes"],
                {"1-class": {"description": "Always output 1-class as the class."}},
            )

    def test_do_not_include_class_names(self):
        sample_detection_dataset, tempdir = DetectionTestFixtures.create_an_od_dataset()
        with tempdir:
            kvp_dataset = DetectionAsKeyValuePairDataset(
                sample_detection_dataset, include_class_names=False
            )

            self.assertIsInstance(kvp_dataset, DetectionAsKeyValuePairDataset)
            self.assertEqual(
                kvp_dataset.dataset_info.schema["fieldSchema"],
                {
                    "detectedObjects": {
                        "type": "array",
                        "description": "Objects in the image of the specified classes, with bounding boxes",
                        "items": {
                            "type": "string",
                            "description": "Class name of the object",
                            "includeGrounding": True,
                        },
                    }
                },
            )
    
    def test_custom_schema_description(self):
        sample_detection_dataset, tempdir = DetectionTestFixtures.create_an_od_dataset()
        with tempdir:
            kvp_dataset = DetectionAsKeyValuePairDataset(
                sample_detection_dataset, custom_schema_description="Custom description"
            )

            self.assertIsInstance(kvp_dataset, DetectionAsKeyValuePairDataset)
            self.assertEqual(
                kvp_dataset.dataset_info.schema["description"], "Custom description"
            )



class TestDetectionAsKeyValuePairDatasetForMultilabelClassification(unittest.TestCase):
    def test_detection_to_kvp(self):
        sample_detection_dataset, tempdir = DetectionTestFixtures.create_an_od_dataset()
        with tempdir:
            kvp_dataset = DetectionAsKeyValuePairDatasetForMultilabelClassification(
                sample_detection_dataset
            )

            self.assertIsInstance(
                kvp_dataset, DetectionAsKeyValuePairDatasetForMultilabelClassification
            )
            self.assertEqual(kvp_dataset.dataset_info.type, DatasetTypes.KEY_VALUE_PAIR)
            self.assertIn("name", kvp_dataset.dataset_info.schema)
            self.assertIn("description", kvp_dataset.dataset_info.schema)
            self.assertIn("fieldSchema", kvp_dataset.dataset_info.schema)

            print(kvp_dataset.dataset_info.schema["fieldSchema"])
            self.assertEqual(
                kvp_dataset.dataset_info.schema["fieldSchema"],
                {
                    "objectClassNames": {
                        "type": "array",
                        "description": "Unique class names of objects in the image of the specified classes.",
                        "items": {
                            "type": "string",
                            "description": "Class name of the object.",
                            "classes": {
                                "1-class": {},
                                "2-class": {},
                                "3-class": {},
                                "4-class": {},
                            },
                        },
                    }
                },
            )
            _, target, _ = kvp_dataset[0]
            print(target.label_data)
            self.assertIsInstance(target, KeyValuePairLabelManifest)
            self.assertEqual(
                target.label_data,
                {
                    "fields": {
                        "objectClassNames": {
                            "value": [{"value": "1-class"}, {"value": "2-class"}]
                        }
                    }
                },
            )

    def test_do_not_include_class_names(self):
        sample_detection_dataset, tempdir = DetectionTestFixtures.create_an_od_dataset()
        with tempdir:
            kvp_dataset = kvp_dataset = (
                DetectionAsKeyValuePairDatasetForMultilabelClassification(
                    sample_detection_dataset, include_class_names=False
                )
            )

            self.assertIsInstance(
                kvp_dataset, DetectionAsKeyValuePairDatasetForMultilabelClassification
            )
            self.assertEqual(
                kvp_dataset.dataset_info.schema["fieldSchema"],
                {
                    "objectClassNames": {
                        "type": "array",
                        "description": "Unique class names of objects in the image of the specified classes.",
                        "items": {
                            "type": "string",
                            "description": "Class name of the object.",
                        },
                    }
                },
            )
    def test_custom_schema_description(self):
        sample_detection_dataset, tempdir = DetectionTestFixtures.create_an_od_dataset()
        with tempdir:
            kvp_dataset = DetectionAsKeyValuePairDatasetForMultilabelClassification(
                sample_detection_dataset, custom_schema_description="Custom description"
            )

            self.assertIsInstance(kvp_dataset, DetectionAsKeyValuePairDatasetForMultilabelClassification)
            self.assertEqual(
                kvp_dataset.dataset_info.schema["description"], "Custom description"
            )


if __name__ == "__main__":
    unittest.main()
