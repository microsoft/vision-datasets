from typing import List
from ..common import ImageLabelManifest


class Grounding:
    def __init__(self, label_data: dict):
        self._label_data = label_data

    @property
    def text(self):
        return self._label_data['text']

    @property
    def bbox(self):
        return self._label_data['bbox']


class VisualObjectGroundingLabelManifest(ImageLabelManifest):
    """
    {"question": "a question about the image",  "answer": "generic caption or answer to the question", "grounding": [{"text": "....", "bbox": [left, top, right, bottom]}, ...]}
    """

    def _read_label_data(self):
        raise NotImplementedError

    def _check_label(self, label_data):
        def is_present(key):
            return key in label_data and label_data[key] is not None

        if label_data is None or any(not is_present(key) for key in ['question', 'answer', 'grounding']):
            raise ValueError

        for grounding in label_data["grounding"]:
            if "text" not in grounding or "bbox" not in grounding or len(grounding['bbox']) != 4:
                raise ValueError

    @property
    def question(self) -> str:
        return self._label_data["question"]

    @property
    def answer(self) -> str:
        return self._label_data["answer"]

    @property
    def grounding(self) -> List[Grounding]:
        return [Grounding(x) for x in self._label_data["grounding"]]