import unittest

from tests.test_fixtures import VQATestFixtures
from vision_datasets.common.constants import DatasetTypes
from vision_datasets.key_value_pair.manifest import KeyValuePairLabelManifest
from vision_datasets.visual_question_answering import VQAAsKeyValuePairDataset


class TestVQAAsKeyValuePairDataset(unittest.TestCase):
    def test_vqa_to_kvp(self):
        sample_vqa_dataset, tempdir = VQATestFixtures.create_a_vqa_dataset()
        with tempdir:
            kvp_dataset = VQAAsKeyValuePairDataset(sample_vqa_dataset)

            self.assertIsInstance(kvp_dataset, VQAAsKeyValuePairDataset)
            self.assertEqual(kvp_dataset.dataset_info.type, DatasetTypes.KEY_VALUE_PAIR)
            self.assertIn("name", kvp_dataset.dataset_info.schema)
            self.assertIn("description", kvp_dataset.dataset_info.schema)
            self.assertIn("fieldSchema", kvp_dataset.dataset_info.schema)

            self.assertEqual(kvp_dataset.dataset_info.schema["fieldSchema"],
                             {'answer': {'type': 'string', 'description': 'Answer to the question.'},
                             'rationale': {'type': 'string', 'description': 'Rationale for the answer.'}})

            _, target, _ = kvp_dataset[0]
            self.assertIsInstance(target, KeyValuePairLabelManifest)
            self.assertEqual(target.label_data,
                             {'fields': {'answer': {'value': 'answer 1'}}, 'text': {'question': 'question 1'}})

            self.assertEqual(len(kvp_dataset), 3)
            self.assertEqual(len(kvp_dataset.dataset_manifest.images), 2)

            # Last image has 2 questions associated with it
            self.assertEqual(kvp_dataset[-2][0][0].size, kvp_dataset[-1][0][0].size)
