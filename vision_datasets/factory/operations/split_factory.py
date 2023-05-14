from ...common import DatasetTypes
from ...data_manifest import Operation, SplitConfig


class SplitFactory:
    _mapping = {}

    @classmethod
    def direct_register(cls, klass, data_type: DatasetTypes):
        cls._mapping[data_type] = klass
        return klass

    @classmethod
    def register(cls, data_type: DatasetTypes):
        def decorator(klass):
            return SplitFactory.direct_register(klass, data_type)
        return decorator

    @classmethod
    def create(cls, data_type: DatasetTypes, config: SplitConfig, *args, **kwargs) -> Operation:

        return cls._mapping[data_type](config, *args, **kwargs)
