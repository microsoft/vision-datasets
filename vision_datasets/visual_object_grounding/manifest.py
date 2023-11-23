from typing import List, Union
from ..common import ImageLabelManifest


class Grounding:
    def __init__(self, label_data: dict, answer_len: int):
        self.check_label(label_data, answer_len)

        self._label_data = label_data

    @staticmethod
    def check_label(label_data, answer_len):
        if not label_data or any(label_data.get(key) is None for key in ['id', 'text_span', 'text', 'bboxes']):
            raise ValueError

        for bbox in label_data['bboxes']:
            if len(bbox) != 4:
                raise ValueError

        if len(label_data['text_span']) != 2:
            raise ValueError

        start = label_data['text_span'][0]
        end = label_data['text_span'][1]
        if start < 0 or end < 0 or start >= end or start >= answer_len or end > answer_len:
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
    def bboxes(self) -> List[List[Union[int, float]]]:
        """ returns a list of bounding boxes in the format of [[left, top, right, bottom], ...]

        Returns:
            List[List[Union[int, float]]]: list of boxes in the format of [[left, top, right, bottom], ...]
        """
        return self._label_data['bboxes']


class VisualObjectGroundingLabelManifest(ImageLabelManifest):
    """
    {
        "question": "a question about the image",
        "answer": "generic caption or answer to the question",
        "groundings": [{"text": "....", "text_span": [start, end], "bboxes": [[left, top, right, bottom], ...]}, ...]
    }
    """

    def _read_label_data(self):
        raise NotImplementedError

    def _check_label(self, label_data):
        if not label_data or any(label_data.get(key, None) is None for key in ['question', 'answer', 'groundings']):
            raise ValueError(str(label_data.keys()))

        for grounding in label_data["groundings"]:
            Grounding.check_label(grounding, len(label_data["answer"]))

    @property
    def question(self) -> str:
        return self.label_data["question"]

    @property
    def answer(self) -> str:
        return self.label_data["answer"]

    @property
    def groundings(self) -> List[Grounding]:
        return [Grounding(x, len(self.label_data["answer"])) for x in self.label_data["groundings"]]
