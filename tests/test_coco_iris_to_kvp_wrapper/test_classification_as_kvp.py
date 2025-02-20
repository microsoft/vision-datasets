import unittest

from tests.test_fixtures import MulticlassClassificationTestFixtures, MultilabelClassificationTestFixtures
from vision_datasets.common import DatasetTypes
from vision_datasets.image_classification import MulticlassClassificationAsKeyValuePairDataset, MultilabelClassificationAsKeyValuePairDataset
from vision_datasets.key_value_pair.manifest import KeyValuePairLabelManifest


class TestClassificationAsKeyValuePairDataset(unittest.TestCase):
    def test_multiclass_classification(self):
        sample_classification_dataset, tempdir = MulticlassClassificationTestFixtures.create_an_ic_dataset()
        with tempdir:
            kvp_dataset = MulticlassClassificationAsKeyValuePairDataset(sample_classification_dataset)

            self.assertIsInstance(kvp_dataset, MulticlassClassificationAsKeyValuePairDataset)
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
        sample_classification_dataset, tempdir = MultilabelClassificationTestFixtures.create_an_ic_dataset(n_images=2, n_categories=2)
        with tempdir:
            kvp_dataset = MultilabelClassificationAsKeyValuePairDataset(sample_classification_dataset)

            self.assertIsInstance(kvp_dataset, MultilabelClassificationAsKeyValuePairDataset)
            self.assertEqual(kvp_dataset.dataset_info.type, DatasetTypes.KEY_VALUE_PAIR)
            self.assertIn("name", kvp_dataset.dataset_info.schema)
            self.assertIn("description", kvp_dataset.dataset_info.schema)
            self.assertIn("fieldSchema", kvp_dataset.dataset_info.schema)

            self.assertEqual(kvp_dataset.dataset_info.schema["fieldSchema"],
                             {'classNames': {
                                 'type': 'array',
                                 'description': 'Class names that the image belongs to.',
                                 'items': {
                                     'type': 'string',
                                     'description': 'Single class name.',
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
                                 'classNames': {'value': [{'value': '1-class'}, {'value': '2-class'}]}}
                              })


if __name__ == '__main__':
    unittest.main()
