import unittest

from tests.test_fixtures import DetectionTestFixtures
from vision_datasets.image_object_detection.detection_as_kvp_dataset import (
    DetectionAsKeyValuePairDataset,
)


class TestClassificationAsKeyValuePairDataset(unittest.TestCase):
    def test_simple(self):
        sample_detection_dataset, _ = DetectionTestFixtures.create_an_od_dataset()
        kvp_dataset = DetectionAsKeyValuePairDataset(sample_detection_dataset)

        print(kvp_dataset)


if __name__ == '__main__':
    unittest.main()
