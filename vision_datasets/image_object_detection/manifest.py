

from ..common import ImageLabelWithCategoryManifest


class ImageObjectDetectionLabelManifest(ImageLabelWithCategoryManifest):
    """
    [c_id, left, top, right, bottom], ...] (absolute coordinates);
    """

    @property
    def category_id(self) -> int:
        return self.label_data[0]

    @property
    def left(self):
        return self.label_data[1]

    @property
    def top(self):
        return self.label_data[2]

    @property
    def right(self):
        return self.label_data[3]

    @property
    def bottom(self):
        return self.label_data[4]

    @category_id.setter
    def category_id(self, value):
        self._category_id_check(value)
        self.label_data[0] = value

    def _read_label_data(self):
        raise NotImplementedError

    def _check_label(self, label_data):
        if not label_data or len(label_data) != 5:
            raise ValueError
