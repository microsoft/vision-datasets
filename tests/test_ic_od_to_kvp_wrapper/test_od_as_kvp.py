import unittest

from tests.test_fixtures import DetectionTestFixtures
from vision_datasets.common.constants import DatasetTypes
from vision_datasets.image_object_detection import DetectionAsKeyValuePairDataset
from vision_datasets.key_value_pair.manifest import KeyValuePairLabelManifest


class TestDetectionAsKeyValuePairDataset(unittest.TestCase):
    def test_detection_to_kvp(self):
        sample_detection_dataset, _ = DetectionTestFixtures.create_an_od_dataset()
        kvp_dataset = DetectionAsKeyValuePairDataset(sample_detection_dataset)

        self.assertIsInstance(kvp_dataset, DetectionAsKeyValuePairDataset)
        self.assertEqual(kvp_dataset.dataset_info.type, DatasetTypes.KEY_VALUE_PAIR)
        self.assertIn("name", kvp_dataset.dataset_info.schema)
        self.assertIn("description", kvp_dataset.dataset_info.schema)
        self.assertIn("fieldSchema", kvp_dataset.dataset_info.schema)

        self.assertEqual(kvp_dataset.dataset_info.schema["fieldSchema"],
                         {'detectedObjects': {'type': 'array', 'description': 'Objects in the image of the specified classes, with bounding boxes',
                                              'items': {'type': 'string', 'description': 'Class name of the object',
                                                        'classes': {'1-class': {'description': 'A single class name. Only output 1-class as the class name if present.'},
                                                                    '2-class': {'description': 'A single class name. Only output 2-class as the class name if present.'},
                                                                    '3-class': {'description': 'A single class name. Only output 3-class as the class name if present.'},
                                                                    '4-class': {'description': 'A single class name. Only output 4-class as the class name if present.'}},
                                                        'includeGrounding': True}}})

        _, target, _ = kvp_dataset[0]
        self.assertIsInstance(target, KeyValuePairLabelManifest)
        self.assertEqual(target.label_data,
                         {'fields': {'detectedObjects': {'value': [{'value': '1-class', 'groundings': [[0, 0, 100, 100]]},
                                                                   {'value': '2-class', 'groundings': [[10, 10, 50, 100]]}]}}
                          })


if __name__ == '__main__':
    unittest.main()
