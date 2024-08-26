import unittest

from vision_datasets.key_value_pair.manifest import KeyValuePairLabelManifest


class TestKeyValuePair(unittest.TestCase):
    id = 1
    img_ids = [1, 2]

    def test_simple(self):
        manifest = KeyValuePairLabelManifest(self.id, self.img_ids, {'fields': {'key1': 'val1'}, 'text': {'query': 'what fields are there?'}})
        self.assertEqual(manifest.id, self.id)
        self.assertEqual(manifest.img_ids, self.img_ids)
        self.assertEqual(manifest.fields, {'key1': 'val1'})
        self.assertEqual(manifest.text, {'query': 'what fields are there?'})

    def test_missing_key_value_pair(self):
        with self.assertRaises(ValueError):
            KeyValuePairLabelManifest(self.id, self.img_ids, {'text': {'query': 'what fields are there?'}})
            
    def test_missing_text(self):
        manifest = KeyValuePairLabelManifest(self.id, self.img_ids, {'fields': {'key1': 'val1'}})
        self.assertIsNone(manifest.text)
