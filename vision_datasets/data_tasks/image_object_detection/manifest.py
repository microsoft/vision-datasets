import pathlib
import typing

from ...data_manifest import ImageLabelWithCategoryManifest


class ImageObjectDetectionLabelManifest(ImageLabelWithCategoryManifest):
    """
    [c_id, left, top, right, bottom], ...] (absolute coordinates);
    """

    def __init__(self, label: typing.List[int], label_path: pathlib.Path = None, additional_info: typing.Dict = None):
        assert label and len(label) == 5
        super().__init__(label, label_path, additional_info)

    @property
    def category_id(self):
        return self.label_data[0]

    @category_id.setter
    def category_id(self, value):
        self._category_id_check(value)
        self.label_data[0] = value
