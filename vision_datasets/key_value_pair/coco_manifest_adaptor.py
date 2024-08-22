import logging
from ..common import BBoxFormat, DatasetTypes, CocoManifestAdaptorFactory, CocoManifestWithMultiImageLabelAdaptor
from .manifest import KeyValuePairLabelManifest, KeyValuePairDatasetManifest, KeyValuePairSchema

logger = logging.getLogger(__name__)


@CocoManifestAdaptorFactory.register(DatasetTypes.KEY_VALUE_PAIR)
class KeyValuePairCocoManifestAdaptor(CocoManifestWithMultiImageLabelAdaptor):
    def __init__(self, schema: dict) -> None:
        self.schema_dict = schema
        self.schema = KeyValuePairSchema(schema['name'], schema['fieldSchema'], schema.get('description', None))
        super().__init__(DatasetTypes.KEY_VALUE_PAIR)
    
    def _construct_label_manifest(self, img_ids, ann, coco_manifest):
        label_data = self.process_label(ann, coco_manifest)
        return KeyValuePairLabelManifest(ann['id'], img_ids, label_data, self._get_additional_info(ann, {'id', KeyValuePairLabelManifest.IMAGES_INPUT_KEY, KeyValuePairLabelManifest.LABEL_KEY,
                                                                                                         KeyValuePairLabelManifest.TEXT_INPUT_KEY}))

    def _construct_manifest(self, images_by_id, coco_manifest, data_type, additional_info):
        images, annotations = self.get_images_and_annotations(images_by_id, coco_manifest)
        return KeyValuePairDatasetManifest(images, annotations, self.schema_dict, additional_info)

    def convert_bbox_ltwh_to_ltrb(self, value):
        if isinstance(value, list):
            for i in range(len(value)):
                self.convert_bbox_ltwh_to_ltrb(value[i])
        elif isinstance(value, dict):
            for k in value.keys():
                if k == KeyValuePairLabelManifest.LABEL_GROUNDINGS_KEY:
                    for i, grounding in enumerate(value[k]):
                        value[k][i] = [grounding[0], grounding[1], grounding[0] + grounding[2], grounding[1] + grounding[3]]
                else:
                    self.convert_bbox_ltwh_to_ltrb(value[k])

    def check_no_groundings_for_multi_image_annotation(self, value: dict):
        if isinstance(value, list):
            for v in value:
                self.check_no_groundings_for_multi_image_annotation(v)
        elif isinstance(value, dict):
            if KeyValuePairLabelManifest.LABEL_GROUNDINGS_KEY in value:
                raise ValueError('Groundings are not allowed for multi-image annotations')            
            for v in value.values():
                self.check_no_groundings_for_multi_image_annotation(v)
                
    def process_label(self, annotation: dict, coco_manifest: dict):
        if KeyValuePairLabelManifest.LABEL_KEY not in annotation:
            raise ValueError(f'{KeyValuePairLabelManifest.LABEL_KEY} not found in annotation {annotation}')            
        
        bbox_format = coco_manifest.get('bbox_format')
        bbox_format = BBoxFormat[bbox_format.upper()] if bbox_format else BBoxFormat.LTWH
        if bbox_format == BBoxFormat.LTWH:
            logger.info('Provided bounding box format is LTWH, converting bounding boxes (if any) to LTRB')
            for k in annotation[KeyValuePairLabelManifest.LABEL_KEY]:
                self.convert_bbox_ltwh_to_ltrb(annotation[KeyValuePairLabelManifest.LABEL_KEY][k])

        # If the annotation is for multiple images, groundings should be disabled
        if len(annotation['image_ids']) > 1:
            print('hereere')
            for field in annotation[KeyValuePairLabelManifest.LABEL_KEY].values():
                self.check_no_groundings_for_multi_image_annotation(field)

        label_data = {KeyValuePairLabelManifest.LABEL_KEY: annotation[KeyValuePairLabelManifest.LABEL_KEY], 
                      KeyValuePairLabelManifest.TEXT_INPUT_KEY: annotation.get(KeyValuePairLabelManifest.TEXT_INPUT_KEY, None)}
        KeyValuePairLabelManifest.check_schema_match(label_data[KeyValuePairLabelManifest.LABEL_KEY], self.schema)
        return label_data
