import logging
from typing import Union

from ..common import DatasetTypes, CocoManifestAdaptorFactory, CocoManifestWithMultiImageAnnotationAdaptor, AnnotationDataManifest
from .manifest import KVPairLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.KV_PAIR)
class KVPairCocoManifestAdaptor(CocoManifestWithMultiImageAnnotationAdaptor):
    def __init__(self, schema: dict) -> None:
        self._schema = schema
        super().__init__(DatasetTypes.KV_PAIR)

    def _schema_match(self, key_value_pairs: dict , field_schema: dict) -> bool:
        for key, value in field_schema.items():
            if key not in key_value_pairs:
                return False
            # TODO: add value type check after we finalize schema defnition
        return True
    
    def process_label(self, ann_manifest: AnnotationDataManifest, annotation: dict, coco_manifest: dict):
        if not self._schema_match(annotation['key_value_pairs'], self._schema['fieldSchema']):
            raise ValueError(f"Annotation key_value_pairs does not match schema: {annotation['key_value_pairs']}")
        ann_manifest.labels.append(KVPairLabelManifest(label_data={'key_value_pairs': annotation['key_value_pairs'], 'text_input': annotation.get('text_input', None)}))
