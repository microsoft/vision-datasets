import copy
from ..common import BBoxFormat, DatasetTypes, ImageDataManifest, CocoManifestWithoutCategoriesAdaptor, CocoManifestAdaptorFactory
from .manifest import VisualObjectGroundingLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.VISUAL_OBJECT_GROUNDING)
class VisualObjectGroundingCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.VISUAL_OBJECT_GROUNDING)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        bbox_format = coco_manifest.get('bbox_format')
        bbox_format = BBoxFormat[bbox_format.upper()] if bbox_format else BBoxFormat.LTWH

        groundings = copy.deepcopy(annotation['groundings'])
        if bbox_format != BBoxFormat.LTRB:
            for grounding in groundings:
                boxes = grounding['bboxes']
                grounding['bboxes'] = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes]

        image.labels.append(VisualObjectGroundingLabelManifest({'question': annotation['question'], 'answer': annotation['answer'], 'groundings': groundings},
                                                               additional_info=self._get_additional_info(annotation, {'id', 'question', 'answer', 'groundings'})))
