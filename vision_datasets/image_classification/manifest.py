from ..common import ImageLabelWithCategoryManifest


class ImageClassificationLabelManifest(ImageLabelWithCategoryManifest):
    """
    c_id: class id starting from zero
    """

    @property
    def category_id(self):
        return self.label_data

    @category_id.setter
    def category_id(self, value):
        self._category_id_check(value)
        self.label_data = value

    def _read_label_data(self):
        raise NotImplementedError

    def _check_label(self, label_data):
        if label_data is None or label_data < 0:
            raise ValueError
