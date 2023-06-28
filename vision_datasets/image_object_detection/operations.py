from ..common import DatasetTypes, BalancedInstanceWeightsGenerator, GenerateCocoDictBase, SampleByNumSamples, SampleFewShot, SampleStrategyType, \
    SingleTaskMerge, Spawn, Split, BalancedInstanceWeightsFactory, CocoDictGeneratorFactory, ManifestMergeStrategyFactory, SampleStrategyFactory, SpawnFactory, SplitFactory
from .manifest import ImageObjectDetectionLabelManifest

_DATA_TYPE = DatasetTypes.IMAGE_OBJECT_DETECTION


@CocoDictGeneratorFactory.register(_DATA_TYPE)
class ImageObjectDetectionCocoDictGenerator(GenerateCocoDictBase):
    def process_labels(self, coco_ann, label: ImageObjectDetectionLabelManifest):
        ann = label.label_data
        coco_ann['category_id'] = ann[0] + 1
        coco_ann['bbox'] = [ann[1], ann[2], ann[3] - ann[1], ann[4] - ann[2]]


ManifestMergeStrategyFactory.direct_register(SingleTaskMerge, _DATA_TYPE)


SampleStrategyFactory.direct_register(SampleByNumSamples, _DATA_TYPE, SampleStrategyType.NumSamples)
SampleStrategyFactory.direct_register(SampleFewShot, _DATA_TYPE, SampleStrategyType.FewShot)


BalancedInstanceWeightsFactory.direct_register(BalancedInstanceWeightsGenerator, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)

SpawnFactory.direct_register(Spawn, _DATA_TYPE)
SplitFactory.direct_register(Split, _DATA_TYPE)
