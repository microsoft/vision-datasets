from ...constants import DatasetTypes


class SupportedOperationsByDataType:
    _mapping = {}

    @classmethod
    def add(cls, data_type: DatasetTypes, klass):
        ops = cls._mapping.setdefault(data_type, [])
        ops.append(klass)

    @classmethod
    def list(cls, data_type: DatasetTypes):
        return cls._mapping.get(data_type, [])
