import abc

from ..data_manifest import DatasetManifest


class Operation(abc.ABC):
    def __init__(self) -> None:
        pass

    def run(*args: DatasetManifest):
        pass
