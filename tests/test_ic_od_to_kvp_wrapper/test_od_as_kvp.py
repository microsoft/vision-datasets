import unittest

from tests.test_fixtures import DetectionTestFixtures
from vision_datasets.common.constants import DatasetTypes
from vision_datasets.image_object_detection.detection_as_kvp_dataset import (
    DetectionAsKeyValuePairDataset,
)
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
                         {'bboxes': {'type': 'array', 'description': 'Bounding boxes of objects in the image',
                                     'items': {'type': 'string', 'description': 'Class name that the box belongs to',
                                               'classes': {'1-class': {},
                                                           '2-class': {},
                                                           '3-class': {},
                                                           '4-class': {}},
                                               'includeGrounding': True}}})

        _, target, _ = kvp_dataset[0]
        self.assertIsInstance(target, KeyValuePairLabelManifest)
        self.assertEqual(target.label_data,
                         {'fields': {'bboxes': {'value': [{'value': '0', 'groundings': [[0, 0, 100, 100]]},
                                                          {'value': '1', 'groundings': [[10, 10, 50, 100]]}]}},
                          'text': None})


if __name__ == '__main__':
    unittest.main()
