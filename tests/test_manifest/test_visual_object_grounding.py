import unittest

from vision_datasets.visual_object_grounding.manifest import VisualObjectGroundingLabelManifest


class TestVisualObjectGrounding(unittest.TestCase):
    def test_simple(self):
        manifest = VisualObjectGroundingLabelManifest({'question': 'hello', 'answer': 'world', 'groundings': [{'id': 1, 'text_span': [0, 2], 'text': 'he', 'bboxes': [[0, 0, 10, 10]]}]})
        self.assertEqual(manifest.question, 'hello')
        self.assertEqual(manifest.answer, 'world')
        self.assertEqual(manifest.groundings[0].id, 1)
        self.assertEqual(manifest.groundings[0].text_span, [0, 2])
        self.assertEqual(manifest.groundings[0].text, 'he')
        self.assertEqual(manifest.groundings[0].bboxes, [[0, 0, 10, 10]])
