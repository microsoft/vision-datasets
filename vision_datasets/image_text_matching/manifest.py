from ..common import ImageLabelManifest


class ImageTextMatchingLabelManifest(ImageLabelManifest):
    """
    (text, match): where text is str, and match in [0, 1]
    """

    def _read_label_data(self):
        raise NotImplementedError

    def _check_label(self, label_data):
        if not label_data or len(label_data) != 2 or label_data[0] is None or label_data[1] not in [0, 1]:
            raise ValueError
