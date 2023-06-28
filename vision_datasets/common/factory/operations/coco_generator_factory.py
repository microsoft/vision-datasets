from ...constants import DatasetTypes
from ...data_manifest import GenerateCocoDictBase
from .supported_operations_by_data_type import SupportedOperationsByDataType


class CocoDictGeneratorFactory:
    _mapping = {}

    @classmethod
    def register(cls, data_type: DatasetTypes):
        def decorator(klass):
            cls._mapping[data_type] = klass
            SupportedOperationsByDataType.add(data_type, klass)
            return klass
        return decorator

    @classmethod
    def create(cls, data_type: DatasetTypes, *args, **kwargs) -> GenerateCocoDictBase:
        return cls._mapping[data_type](*args, **kwargs)
