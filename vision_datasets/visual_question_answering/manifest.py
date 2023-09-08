from ..common import ImageLabelManifest


class VisualQuestionAnsweringLabelManifest(ImageLabelManifest):
    """
    {"question": "a question about the image",  "answer": "answer to the question"}
    """

    def _read_label_data(self):
        raise NotImplementedError

    def _check_label(self, label_data):
        if label_data is None or 'question' not in label_data or 'answer' not in label_data:
            raise ValueError

    @property
    def question(self) -> str:
        return self.label_data["question"]

    @property
    def answer(self) -> str:
        return self.label_data["answer"]
