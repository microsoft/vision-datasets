import unittest

from vision_datasets.kv_pair.manifest import KVPairLabelManifest


class TestKVPair(unittest.TestCase):
    def test_simple(self):
        manifest = KVPairLabelManifest({'key_value_pairs': {'key1': 'val1'}, 'text_input': {'query': 'what fields are there?'}})
        self.assertEqual(manifest.key_value_pairs, {'key1': 'val1'})
        self.assertEqual(manifest.text_input, {'query': 'what fields are there?'})

    def test_missing_kv_pair(self):
        with self.assertRaises(ValueError):
            KVPairLabelManifest({'text_input': {'query': 'what fields are there?'}})
            
    def test_missing_text_input(self):
        manifest = KVPairLabelManifest({'key_value_pairs': {'key1': 'val1'}})
        self.assertIsNone(manifest.text_input)
