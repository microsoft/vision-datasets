from ..common import DatasetTypes, CocoManifestAdaptorFactory, CocoManifestWithMultiImageAnnotationAdaptor, AnnotationDataManifest
from .manifest import KVPairAnnotationManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.KV_PAIR)
class KVPairCocoManifestAdaptor(CocoManifestWithMultiImageAnnotationAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.KV_PAIR)

    def process_label(self, ann_manifest: AnnotationDataManifest, annotation: dict, coco_manifest: dict):
        ann_manifest.labels.append(KVPairAnnotationManifest({'key_value_pairs': annotation['key_value_pairs'], 'text_input': annotation.get('text_input', None)}))
