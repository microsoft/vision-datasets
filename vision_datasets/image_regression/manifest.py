from ..common import ImageLabelManifest


class ImageRegressionLabelManifest(ImageLabelManifest):
    """
    value: regression target in float
    """

    def _read_label_data(self):
        raise NotImplementedError

    def _check_label(self, label_data):
        if label_data is None:
            raise ValueError
