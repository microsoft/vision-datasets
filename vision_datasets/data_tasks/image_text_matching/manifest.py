import pathlib
import typing

from ...data_manifest import ImageLabelManifest


class ImageTextMatchingLabelManifest(ImageLabelManifest):
    """
    (text, match): where text is str, and match in [0, 1]
    """

    def __init__(self, label: typing.Tuple[str, int], label_path: pathlib.Path = None, additional_info: typing.Dict = None):
        assert label[0]
        assert label[1] in [0, 1]

        super().__init__(label, label_path, additional_info)

    def _read_label_data(self):
        raise NotImplementedError
