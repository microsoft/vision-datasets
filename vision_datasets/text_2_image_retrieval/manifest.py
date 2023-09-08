from ..common import ImageLabelManifest


class Text2ImageRetrievalLabelManifest(ImageLabelManifest):
    """
    query: in str
    """
    def query(self) -> str:
        return self.label_data

    def _read_label_data(self):
        raise NotImplementedError

    def _check_label(self, label_data):
        if label_data is None:
            raise ValueError
