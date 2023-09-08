from ..common import ImageLabelManifest


class ImageTextMatchingLabelManifest(ImageLabelManifest):
    """
    (text, match): where text is str, and match is between [0, 1], where 0 means not match at all, 1 means perfect match
    """

    @property
    def text(self) -> str:
        return self.label_data[0]

    @property
    def match(self) -> float:
        return self.label_data[1]

    def _read_label_data(self):
        raise NotImplementedError

    def _check_label(self, label_data):
        if not label_data or len(label_data) != 2 or label_data[0] is None or label_data[1] not in [0, 1]:
            raise ValueError
