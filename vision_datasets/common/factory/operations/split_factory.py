import typing
from ...constants import DatasetTypes
from ...data_manifest import Operation, SplitConfig
from .supported_operations_by_data_type import SupportedOperationsByDataType


class SplitFactory:
    _mapping = {}

    @classmethod
    def direct_register(cls, klass, data_type: DatasetTypes):
        cls._mapping[data_type] = klass
        SupportedOperationsByDataType.add(data_type, klass)
        return klass

    @classmethod
    def register(cls, data_type: DatasetTypes):
        def decorator(klass):
            return SplitFactory.direct_register(klass, data_type)
        return decorator

    @classmethod
    def create(cls, data_type: DatasetTypes, config: SplitConfig, *args, **kwargs) -> Operation:

        return cls._mapping[data_type](config, *args, **kwargs)

    @classmethod
    def list_data_types(cls) -> typing.Iterable[DatasetTypes]:
        return list(cls._mapping.keys())
