import unittest

from tests.test_fixtures import MultilcassClassificationTestFixtures
from vision_datasets.common import DatasetTypes
from vision_datasets.image_classification.classification_as_kvp_dataset import (
    ClassificationAsKeyValuePairDataset,
)
from vision_datasets.key_value_pair.manifest import KeyValuePairLabelManifest


class TestClassificationAsKeyValuePairDataset(unittest.TestCase):
    def test_multiclass_classification(self):
        sample_classification_dataset, _ = MultilcassClassificationTestFixtures.create_an_ic_dataset()
        kvp_dataset = ClassificationAsKeyValuePairDataset(sample_classification_dataset)

        self.assertIsInstance(kvp_dataset, ClassificationAsKeyValuePairDataset)
        self.assertEqual(kvp_dataset.dataset_info.type, DatasetTypes.KEY_VALUE_PAIR)
        self.assertIn("name", kvp_dataset.dataset_info.schema)
        self.assertIn("description", kvp_dataset.dataset_info.schema)
        self.assertIn("fieldSchema", kvp_dataset.dataset_info.schema)

        self.assertEqual(kvp_dataset.dataset_info.schema["fieldSchema"],
                         {"className": {
                             "type": "string",
                             "description": "Class name that the image belongs to.",
                             "classes": {
                                 "1-class": {},
                                 "2-class": {},
                                 "3-class": {},
                             }
                         }
        })

        _, target, _ = kvp_dataset[0]
        self.assertIsInstance(target, KeyValuePairLabelManifest)


if __name__ == '__main__':
    unittest.main()
