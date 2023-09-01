import typing
from ...constants import DatasetTypes
from ...data_manifest import SampleBaseConfig, SampleStrategy, SampleStrategyType
from .supported_operations_by_data_type import SupportedOperationsByDataType


class SampleStrategyFactory:
    _mapping = {}

    @classmethod
    def direct_register(cls, klass, data_type: DatasetTypes, strategy_name: SampleStrategyType):
        cls._mapping[(data_type, strategy_name)] = klass
        return klass

    @classmethod
    def register(cls, data_type: DatasetTypes, strategy_name: SampleStrategyType):
        def decorator(klass):
            SupportedOperationsByDataType.add(data_type, klass)
            return SampleStrategyFactory.direct_register(klass, data_type, strategy_name)
        return decorator

    @classmethod
    def create(cls, data_type: DatasetTypes, strategy_type: SampleStrategyType, config: SampleBaseConfig, *args, **kwargs) -> SampleStrategy:
        return cls._mapping[(data_type, strategy_type)](config, *args, **kwargs)

    @classmethod
    def list_data_types(cls, strategy_type: SampleStrategyType) -> typing.Iterable[DatasetTypes]:
        for key in cls._mapping:
            if key[1] == strategy_type:
                yield key[0]
