import abc

from ..data_manifest import DatasetManifest


class Operation(abc.ABC):
    """
    Base class for operations on DatasetManifest
    """

    def __init__(self) -> None:
        pass

    def run(*args: DatasetManifest):
        pass
