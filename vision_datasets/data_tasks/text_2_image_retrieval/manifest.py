import pathlib
import typing

from ...data_manifest import ImageLabelManifest


class Text2ImageRetrievalLabelManifest(ImageLabelManifest):
    """
    query: in str
    """

    def __init__(self, label: str, label_path: pathlib.Path = None, additional_info: typing.Dict = None):
        assert label is not None
        super().__init__(label, label_path, additional_info)
