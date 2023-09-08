from ..common import DatasetTypes, ImageDataManifest, CocoManifestWithoutCategoriesAdaptor, CocoManifestAdaptorFactory
from .manifest import VisualObjectGroundingLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.VISUAL_OBJECT_GROUNDING)
class VisualObjectGroundingCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.VISUAL_OBJECT_GROUNDING)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        image.labels.append(VisualObjectGroundingLabelManifest({'question': annotation['question'], 'answer': annotation['answer'], 'groundings': annotation['groundings']},
                                                               additional_info=self._get_additional_info(annotation, {'id', 'question', 'answer', 'groundings'})))
