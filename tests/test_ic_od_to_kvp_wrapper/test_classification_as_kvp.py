import unittest

from tests.test_fixtures import MulticlassClassificationTestFixtures, MultilabelClassificationTestFixtures
from vision_datasets.common import DatasetTypes
from vision_datasets.image_classification import MultiClassAsKeyValuePairDataset, MultiLabelAsKeyValuePairDataset
from vision_datasets.key_value_pair.manifest import KeyValuePairLabelManifest


class TestClassificationAsKeyValuePairDataset(unittest.TestCase):
    def test_multiclass_classification(self):
        sample_classification_dataset, _ = MulticlassClassificationTestFixtures.create_an_ic_dataset()
        kvp_dataset = MultiClassAsKeyValuePairDataset(sample_classification_dataset)

        self.assertIsInstance(kvp_dataset, MultiClassAsKeyValuePairDataset)
        self.assertEqual(kvp_dataset.dataset_info.type, DatasetTypes.KEY_VALUE_PAIR)
        self.assertIn("name", kvp_dataset.dataset_info.schema)
        self.assertIn("description", kvp_dataset.dataset_info.schema)
        self.assertIn("fieldSchema", kvp_dataset.dataset_info.schema)

        self.assertEqual(kvp_dataset.dataset_info.schema["fieldSchema"],
                         {"className": {
                             "type": "string",
                             "description": "Class name that the image belongs to.",
                             "classes": {
                                 "1-class": {"description": "A single class name. Only output 1-class as the class name if present."},
                                 "2-class": {"description": "A single class name. Only output 2-class as the class name if present."},
                                 "3-class": {"description": "A single class name. Only output 3-class as the class name if present."},
                             }
                         }
        })

        _, target, _ = kvp_dataset[0]
        self.assertIsInstance(target, KeyValuePairLabelManifest)
        self.assertEqual(target.label_data,
                         {"fields": {"className": {"value": "1-class"}}}
                         )

    def test_multilabel_classification(self):
        sample_classification_dataset, _ = MultilabelClassificationTestFixtures.create_an_ic_dataset(n_images=2, n_categories=2)
        kvp_dataset = MultiLabelAsKeyValuePairDataset(sample_classification_dataset)

        self.assertIsInstance(kvp_dataset, MultiLabelAsKeyValuePairDataset)
        self.assertEqual(kvp_dataset.dataset_info.type, DatasetTypes.KEY_VALUE_PAIR)
        self.assertIn("name", kvp_dataset.dataset_info.schema)
        self.assertIn("description", kvp_dataset.dataset_info.schema)
        self.assertIn("fieldSchema", kvp_dataset.dataset_info.schema)

        self.assertEqual(kvp_dataset.dataset_info.schema["fieldSchema"],
                         {'classNames': {
                             'type': 'array',
                             'items': {
                                 'type': 'string',
                                 'description': 'Class names that the image belongs to.',
                                 'classes': {
                                     '1-class': {'description': 'A single class name. Only output 1-class as the class name if present.'},
                                     '2-class': {'description': 'A single class name. Only output 2-class as the class name if present.'}
                                  }
                                 }
                             }
                          }
                         )

        _, target, _ = kvp_dataset[0]
        self.assertIsInstance(target, KeyValuePairLabelManifest)
        self.assertEqual(target.label_data,
                         {'fields': {
                             'classNames': {'value': ['1-class', '2-class']}}
                          })


if __name__ == '__main__':
    unittest.main()
