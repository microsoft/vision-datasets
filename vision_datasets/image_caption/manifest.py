from ..common import ImageLabelManifest


class ImageCaptionLabelManifest(ImageLabelManifest):
    """
    caption: in str
    """

    @property
    def caption(self):
        return self.label_data

    def _read_label_data(self):
        raise NotImplementedError

    def _check_label(self, label_data):
        if label_data is None:
            raise ValueError
