from ..common import DatasetTypes, CocoManifestAdaptorFactory, CocoManifestWithMultiImageAnnotationAdaptor, AnnotationDataManifest
from .manifest import KVPairLabelManifest, KVPairDatasetManifest, Schema


@CocoManifestAdaptorFactory.register(DatasetTypes.KV_PAIR)
class KVPairCocoManifestAdaptor(CocoManifestWithMultiImageAnnotationAdaptor):
    def __init__(self, schema: dict) -> None:
        self.schema_dict = schema
        self.schema = Schema(**schema)
        super().__init__(DatasetTypes.KV_PAIR)
    
    def _construct_manifest(self, images_by_id, coco_manifest, data_type, additional_info):
        additional_info['schema'] = self.schema_dict
        images, annotations = self.get_images_and_annotations(images_by_id, coco_manifest)
        return KVPairDatasetManifest(images, annotations, additional_info)

    def process_label(self, ann_manifest: AnnotationDataManifest, annotation: dict, coco_manifest: dict):
        if KVPairLabelManifest.KV_PAIR_KEY not in annotation:
            raise ValueError(f'{KVPairLabelManifest.KV_PAIR_KEY} not found in annotation {annotation}')            
        
        ann_manifest.label = KVPairLabelManifest(
            label_data={KVPairLabelManifest.KV_PAIR_KEY: annotation[KVPairLabelManifest.KV_PAIR_KEY], 
                        KVPairLabelManifest.INPUT_KEY: annotation.get(KVPairLabelManifest.INPUT_KEY, None)},
            additional_info=self._get_additional_info(annotation, {'id', 'image_id', KVPairLabelManifest.KV_PAIR_KEY, KVPairLabelManifest.INPUT_KEY}))

        try:
            ann_manifest.label.check_schema_match(self.schema)
        except ValueError as e:
            raise ValueError(f'annotation mismatch schema: {e}. Annotation: {annotation}, Schema: {self.schema}')
