from ..common import ImageLabelManifest


class ImageRegressionLabelManifest(ImageLabelManifest):
    """
    value: regression target in float
    """

    @property
    def target(self) -> float:
        return self.label_data

    def _read_label_data(self):
        raise NotImplementedError

    def _check_label(self, label_data):
        if label_data is None:
            raise ValueError
