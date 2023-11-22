from ..common import DatasetTypes, ImageDataManifest, CocoManifestWithoutCategoriesAdaptor, CocoManifestAdaptorFactory, BBoxFormat
from .manifest import VisualObjectGroundingLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.VISUAL_OBJECT_GROUNDING)
class VisualObjectGroundingCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.VISUAL_OBJECT_GROUNDING)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        bbox_format = coco_manifest.get('bbox_format')
        bbox_format = BBoxFormat[bbox_format.upper()] if bbox_format else BBoxFormat.LTWH
        groundings = annotation['groundings']
        if bbox_format == BBoxFormat.LTWH:
            for g in groundings:
                for bbox in g['bboxes']:
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]

        image.labels.append(VisualObjectGroundingLabelManifest({'question': annotation['question'], 'answer': annotation['answer'], 'groundings': annotation['groundings']},
                                                               additional_info=self._get_additional_info(annotation, {'id', 'question', 'answer', 'groundings'})))
