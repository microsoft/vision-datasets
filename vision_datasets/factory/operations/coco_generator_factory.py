from ...common import DatasetTypes
from ...data_manifest.operations import GenerateCocoDictBase


class CocoDictGeneratorFactory:
    _mapping = {}

    @classmethod
    def register(cls, data_type: DatasetTypes):
        def decorator(klass):
            cls._mapping[data_type] = klass
            return klass
        return decorator

    @classmethod
    def create(cls, data_type: DatasetTypes, *args, **kwargs) -> GenerateCocoDictBase:
        return cls._mapping[data_type](*args, **kwargs)
