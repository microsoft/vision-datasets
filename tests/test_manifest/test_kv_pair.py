import unittest

from vision_datasets.kv_pair.manifest import KVPairLabelManifest


class TestKVPair(unittest.TestCase):
    id = 1
    img_ids = [1, 2]
    def test_simple(self):
        manifest = KVPairLabelManifest(self.id, self.img_ids, {'key_value_pairs': {'key1': 'val1'}, 'text_input': {'query': 'what fields are there?'}})
        self.assertEqual(manifest.id, self.id)
        self.assertEqual(manifest.img_ids, self.img_ids)
        self.assertEqual(manifest.key_value_pairs, {'key1': 'val1'})
        self.assertEqual(manifest.text_input, {'query': 'what fields are there?'})

    def test_missing_kv_pair(self):
        with self.assertRaises(ValueError):
            KVPairLabelManifest(self.id, self.img_ids, {'text_input': {'query': 'what fields are there?'}})
            
    def test_missing_text_input(self):
        manifest = KVPairLabelManifest(self.id, self.img_ids, {'key_value_pairs': {'key1': 'val1'}})
        self.assertIsNone(manifest.text_input)
