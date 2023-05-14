import pathlib
import typing

from ...data_manifest import ImageLabelWithCategoryManifest


class ImageClassificationLabelManifest(ImageLabelWithCategoryManifest):
    """
    c_id: class id starting from zero
    """

    def __init__(self, label: int, label_path: pathlib.Path = None, additional_info: typing.Dict = None):
        assert label is not None and label >= 0
        super().__init__(label, label_path, additional_info)

    @property
    def category_id(self):
        return self.label_data

    @category_id.setter
    def category_id(self, value):
        self._category_id_check(value)
        self.label_data = value
