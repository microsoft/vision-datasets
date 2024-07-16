from ..common import DatasetTypes, CocoManifestAdaptorFactory, CocoManifestWithMultiImageAnnotationAdaptor, AnnotationDataManifest
from .manifest import KVPairLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.KV_PAIR)
class KVPairCocoManifestAdaptor(CocoManifestWithMultiImageAnnotationAdaptor):
    KV_PAIR_KEY = 'key_value_pairs'
    INPUT_KEY = 'text_input'

    def __init__(self, schema: dict={}) -> None:
        self._schema = schema
        super().__init__(DatasetTypes.KV_PAIR)

    def _match_schema(self, annotation: dict , field_schema: dict) -> bool:
        # Check if all keys in field_schema are present in key_value_pairs
        for key, value in field_schema.items():
            if key not in annotation[KVPairCocoManifestAdaptor.KV_PAIR_KEY]:
                raise ValueError(f'{key} in schema not found in annotation id {annotation["id"]}!')
            # TODO: add value type check after we finalize schema defnition
        return True
    
    def process_label(self, ann_manifest: AnnotationDataManifest, annotation: dict, coco_manifest: dict):
        if KVPairCocoManifestAdaptor.KV_PAIR_KEY not in annotation:
            raise ValueError(f'{KVPairCocoManifestAdaptor.KV_PAIR_KEY} not found in annotation {annotation}')            
        try:
            self._match_schema(annotation, self._schema.get('fieldSchema', {}))
        except ValueError as e:
            raise ValueError(e)
        
        ann_manifest.label = KVPairLabelManifest(
            label_data={KVPairCocoManifestAdaptor.KV_PAIR_KEY: annotation[KVPairCocoManifestAdaptor.KV_PAIR_KEY], 
                        KVPairCocoManifestAdaptor.INPUT_KEY: annotation.get(KVPairCocoManifestAdaptor.INPUT_KEY, None)},
            additional_info=self._get_additional_info(annotation, {'id', 'image_id', KVPairCocoManifestAdaptor.KV_PAIR_KEY, KVPairCocoManifestAdaptor.INPUT_KEY}))
