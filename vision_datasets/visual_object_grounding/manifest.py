from typing import List, Union
from ..common import ImageLabelManifest


class Grounding:
    def __init__(self, label_data: dict):
        self.check_label(label_data)

        self._label_data = label_data

    @staticmethod
    def check_label(label_data, answer_len):
        if not label_data or any(label_data.get(key) is None for key in ['id', 'text_span', 'text', 'bbox']):
            raise ValueError

        if len(label_data['bbox']) != 4:
            raise ValueError

        if len(label_data['text_span']) != 2:
            raise ValueError

        start = label_data['text_span'][0]
        end = label_data['text_span'][1]
        text = label_data['text']
        if start < 0 or end < 0 or start > end or start > len(text) or end > len(text):
            raise ValueError

    @property
    def id(self):
        return self._label_data['id']

    @property
    def text_span(self):
        return self._label_data['text_span']

    @property
    def text(self) -> str:
        return self._label_data['text']

    @property
    def bbox(self) -> List[Union[int, float]]:
        return self._label_data['bbox']


class VisualObjectGroundingLabelManifest(ImageLabelManifest):
    """
    {
        "question": "a question about the image",
        "answer": "generic caption or answer to the question",
        "grounding": [{"text": "....", "text_span": [start, end], "bbox": [left, top, right, bottom]}, ...]
    }
    """

    def _read_label_data(self):
        raise NotImplementedError

    def _check_label(self, label_data):
        if not label_data or any(label_data.get(key, None) is None for key in ['id', 'text_span', 'text', 'bbox']):
            raise ValueError

        for grounding in label_data["grounding"]:
            Grounding.check_label(grounding, len(label_data["answer"]))

    @property
    def question(self) -> str:
        return self._label_data["question"]

    @property
    def answer(self) -> str:
        return self._label_data["answer"]

    @property
    def grounding(self) -> List[Grounding]:
        return [Grounding(x) for x in self._label_data["grounding"]]
