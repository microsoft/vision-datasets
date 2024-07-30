import abc
import typing
from ..data_manifest import DatasetManifest, DatasetManifestWithMultiImageLabel


class Operation(abc.ABC):
    """
    Base class for operations on DatasetManifest
    """

    def __init__(self) -> None:
        pass

    def run(*args: typing.Union[DatasetManifest, DatasetManifestWithMultiImageLabel]):
        pass
