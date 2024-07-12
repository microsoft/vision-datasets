import abc
import typing
from ..data_manifest import DatasetManifest, AnnotationWiseDatasetManifest


class Operation(abc.ABC):
    """
    Base class for operations on DatasetManifest
    """

    def __init__(self) -> None:
        pass

    def run(*args: typing.Union[DatasetManifest, AnnotationWiseDatasetManifest]):
        pass
