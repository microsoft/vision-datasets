import pathlib
import typing

from ...data_manifest import ImageLabelManifest


class ImageRegressionLabelManifest(ImageLabelManifest):
    """
    value: regression target in float
    """

    def __init__(self, label: float, label_path: pathlib.Path = None, additional_info: typing.Dict = None):
        assert label is not None
        super().__init__(label, label_path, additional_info)

    def _read_label_data(self):
        raise NotImplementedError
