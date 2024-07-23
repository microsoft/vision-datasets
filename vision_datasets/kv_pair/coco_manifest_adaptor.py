from ..common import DatasetTypes, CocoManifestAdaptorFactory, CocoManifestWithMultiImageLabelAdaptor
from .manifest import KVPairLabelManifest, KVPairDatasetManifest, KVPairSchema


@CocoManifestAdaptorFactory.register(DatasetTypes.KEY_VALUE_PAIR)
class KVPairCocoManifestAdaptor(CocoManifestWithMultiImageLabelAdaptor):
    def __init__(self, schema: dict) -> None:
        self.schema_dict = schema
        self.schema = KVPairSchema(schema['name'], schema['fieldSchema'], schema.get('description', None))
        super().__init__(DatasetTypes.KEY_VALUE_PAIR)
    
    def _construct_label_manifest(self, img_ids, ann, coco_manifest):
        label_data = self.process_label(ann, coco_manifest)
        return KVPairLabelManifest(ann['id'], img_ids, label_data, self._get_additional_info(ann, {'id', 'image_ids', KVPairLabelManifest.LABEL_KEY, KVPairLabelManifest.INPUT_KEY}))

    def _construct_manifest(self, images_by_id, coco_manifest, data_type, additional_info):
        additional_info['schema'] = self.schema_dict
        images, annotations = self.get_images_and_annotations(images_by_id, coco_manifest)
        return KVPairDatasetManifest(images, annotations, additional_info)

    def process_label(self, annotation: dict, coco_manifest: dict):
        if KVPairLabelManifest.LABEL_KEY not in annotation:
            raise ValueError(f'{KVPairLabelManifest.LABEL_KEY} not found in annotation {annotation}')            
        
        label_data = {KVPairLabelManifest.LABEL_KEY: annotation[KVPairLabelManifest.LABEL_KEY], 
                      KVPairLabelManifest.INPUT_KEY: annotation.get(KVPairLabelManifest.INPUT_KEY, None)}
        KVPairLabelManifest.check_schema_match(label_data[KVPairLabelManifest.LABEL_KEY], self.schema)
        return label_data
