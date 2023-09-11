from ..common import DatasetTypes, GenerateCocoDictBase, ImageLabelManifest, SampleByNumSamples, SampleStrategyType, SingleTaskMerge, Spawn, Split, CocoDictGeneratorFactory, \
    ManifestMergeStrategyFactory, SampleStrategyFactory, SpawnFactory, SplitFactory, StandAloneImageListGeneratorFactory, GenerateStandAloneImageListBase, \
    ImageDataManifest, DatasetManifest
from .manifest import VisualQuestionAnsweringLabelManifest
_DATA_TYPE = DatasetTypes.VISUAL_QUESTION_ANSWERING


@CocoDictGeneratorFactory.register(_DATA_TYPE)
class VisualQuestionAnsweringCocoDictGenerator(GenerateCocoDictBase):
    def process_labels(self, coco_ann, label: ImageLabelManifest):
        for key, val in label.label_data.items():
            coco_ann[key] = val


ManifestMergeStrategyFactory.direct_register(SingleTaskMerge, _DATA_TYPE)


SampleStrategyFactory.direct_register(SampleByNumSamples, _DATA_TYPE, SampleStrategyType.NumSamples)

SpawnFactory.direct_register(Spawn, _DATA_TYPE)
SplitFactory.direct_register(Split, _DATA_TYPE)


@StandAloneImageListGeneratorFactory.register(_DATA_TYPE)
class VisualQuestionAnsweringStandAloneImageListGenerator(GenerateStandAloneImageListBase):
    def _generate_label(self, label: VisualQuestionAnsweringLabelManifest, image: ImageDataManifest, manifest: DatasetManifest) -> dict:
        return {'question': label.question, 'answer': label.answer}
