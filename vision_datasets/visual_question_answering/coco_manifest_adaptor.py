from ..common import DatasetTypes, ImageDataManifest, CocoManifestWithoutCategoriesAdaptor, CocoManifestAdaptorFactory
from .manifest import VisualQuestionAnsweringLabelManifest


@CocoManifestAdaptorFactory.register(DatasetTypes.VISUAL_QUESTION_ANSWERING)
class VisualQuestionAnswerinCocoManifestAdaptor(CocoManifestWithoutCategoriesAdaptor):
    def __init__(self) -> None:
        super().__init__(DatasetTypes.VISUAL_QUESTION_ANSWERING)

    def process_label(self, image: ImageDataManifest, annotation: dict, coco_manifest: dict):
        image.labels.append(VisualQuestionAnsweringLabelManifest({"question": annotation['question'], "answer": annotation['answer']},
                                                                 additional_info=self._get_additional_info(annotation, {'id', 'question', ''})))
