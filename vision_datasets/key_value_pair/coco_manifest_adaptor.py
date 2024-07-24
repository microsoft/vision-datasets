from ..common import DatasetTypes, CocoManifestAdaptorFactory, CocoManifestWithMultiImageLabelAdaptor
from .manifest import KeyValuePairLabelManifest, KeyValuePairDatasetManifest, KeyValuePairSchema


@CocoManifestAdaptorFactory.register(DatasetTypes.KEY_VALUE_PAIR)
class KeyValuePairCocoManifestAdaptor(CocoManifestWithMultiImageLabelAdaptor):
    def __init__(self, schema: dict) -> None:
        self.schema_dict = schema
        self.schema = KeyValuePairSchema(schema['name'], schema['fieldSchema'], schema.get('description', None))
        super().__init__(DatasetTypes.KEY_VALUE_PAIR)
    
    def _construct_label_manifest(self, img_ids, ann, coco_manifest):
        label_data = self.process_label(ann, coco_manifest)
        return KeyValuePairLabelManifest(ann['id'], img_ids, label_data, self._get_additional_info(ann, {'id', 'image_ids', KeyValuePairLabelManifest.LABEL_KEY, KeyValuePairLabelManifest.INPUT_KEY}))

    def _construct_manifest(self, images_by_id, coco_manifest, data_type, additional_info):
        additional_info['schema'] = self.schema_dict
        images, annotations = self.get_images_and_annotations(images_by_id, coco_manifest)
        return KeyValuePairDatasetManifest(images, annotations, additional_info)

    def process_label(self, annotation: dict, coco_manifest: dict):
        if KeyValuePairLabelManifest.LABEL_KEY not in annotation:
            raise ValueError(f'{KeyValuePairLabelManifest.LABEL_KEY} not found in annotation {annotation}')            
        
        label_data = {KeyValuePairLabelManifest.LABEL_KEY: annotation[KeyValuePairLabelManifest.LABEL_KEY], 
                      KeyValuePairLabelManifest.INPUT_KEY: annotation.get(KeyValuePairLabelManifest.INPUT_KEY, None)}
        KeyValuePairLabelManifest.check_schema_match(label_data[KeyValuePairLabelManifest.LABEL_KEY], self.schema)
        return label_data
