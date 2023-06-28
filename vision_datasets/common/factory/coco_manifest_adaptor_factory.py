from ..data_manifest import CocoManifestAdaptorBase


class CocoManifestAdaptorFactory:
    _mapping = {}

    @classmethod
    def register(cls, data_type: str):
        def decorator(klass):
            cls._mapping[data_type] = klass
            return klass
        return decorator

    @classmethod
    def create(cls, data_type: str, *args, **kwargs) -> CocoManifestAdaptorBase:

        return cls._mapping[data_type](*args, **kwargs)
