from ...common import DatasetTypes
from ...data_manifest import MergeStrategy, MergeStrategyType


class ManifestMergeStrategyFactory:
    _mapping = {}

    @classmethod
    def direct_register(cls, klass, data_type: DatasetTypes, strategy_name: MergeStrategyType):
        cls._mapping[(data_type, strategy_name)] = klass
        return klass

    @classmethod
    def register(cls, data_type: DatasetTypes, strategy_name: MergeStrategyType):
        def decorator(klass):
            return ManifestMergeStrategyFactory.direct_register(klass, data_type, strategy_name)
        return decorator

    @classmethod
    def create(cls, data_type: DatasetTypes, strategy_type: MergeStrategyType, *args, **kwargs) -> MergeStrategy:

        return cls._mapping[(data_type, strategy_type)](*args, **kwargs)
