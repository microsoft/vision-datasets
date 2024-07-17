from ..common import DatasetTypes, CocoManifestAdaptorFactory, CocoManifestWithMultiImageAnnotationAdaptor, AnnotationDataManifest
from .manifest import KVPairLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.KV_PAIR)
class KVPairCocoManifestAdaptor(CocoManifestWithMultiImageAnnotationAdaptor):
    def __init__(self, schema: dict={}) -> None:
        self._schema = schema
        super().__init__(DatasetTypes.KV_PAIR)

    def _match_schema(self, annotation: dict , field_schema: dict) -> bool:
        # Check if all keys in field_schema are present in key_value_pairs
        for key, value in field_schema.items():
            if key not in annotation[KVPairLabelManifest.KV_PAIR_KEY]:
                raise ValueError(f'{key} in schema not found in annotation id {annotation["id"]}!')
            # TODO: add value type check after we finalize schema defnition
        return True
    
    def process_label(self, ann_manifest: AnnotationDataManifest, annotation: dict, coco_manifest: dict):
        if KVPairLabelManifest.KV_PAIR_KEY not in annotation:
            raise ValueError(f'{KVPairLabelManifest.KV_PAIR_KEY} not found in annotation {annotation}')            
        try:
            self._match_schema(annotation, self._schema.get('fieldSchema', {}))
        except ValueError as e:
            raise ValueError(e)
        
        ann_manifest.label = KVPairLabelManifest(
            label_data={KVPairLabelManifest.KV_PAIR_KEY: annotation[KVPairLabelManifest.KV_PAIR_KEY], 
                        KVPairLabelManifest.INPUT_KEY: annotation.get(KVPairLabelManifest.INPUT_KEY, None)},
            additional_info=self._get_additional_info(annotation, {'id', 'image_id', KVPairLabelManifest.KV_PAIR_KEY, KVPairLabelManifest.INPUT_KEY}))
