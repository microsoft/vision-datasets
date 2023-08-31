from typing import List
from ..common import ImageLabelManifest


class GroundingAnswer:
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
    {"question": "a question about the image",  "answer": [{"text": " in text", "bbox": [left, top, right, bottom]}, ...]}
    """

    def _read_label_data(self):
        raise NotImplementedError

    def _check_label(self, label_data):
        if label_data is None or "question" not in label_data or "answer" not in label_data:
            raise ValueError

        for ans in label_data["answer"]:
            if "text" not in ans or "bbox" not in ans or len(ans['bbox']) != 4:
                raise ValueError

    @property
    def question(self) -> str:
        return self._label_data["question"]

    @property
    def answer(self) -> List[GroundingAnswer]:
        return [GroundingAnswer(x) for x in self._label_data["answer"]]
