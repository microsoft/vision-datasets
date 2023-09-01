from ..common import DatasetTypes, GenerateCocoDictBase, ImageLabelManifest, SampleByNumSamples, SampleStrategyType, SingleTaskMerge, Spawn, Split, CocoDictGeneratorFactory, \
    ManifestMergeStrategyFactory, SampleStrategyFactory, SpawnFactory, SplitFactory

_DATA_TYPE = DatasetTypes.IMAGE_TEXT_MATCHING


@CocoDictGeneratorFactory.register(_DATA_TYPE)
class ImageTextMatchingCocoDictGenerator(GenerateCocoDictBase):
    def process_labels(self, coco_ann, label: ImageLabelManifest):
        coco_ann['text'] = label.label_data[0]
        coco_ann['match'] = label.label_data[1]


ManifestMergeStrategyFactory.direct_register(SingleTaskMerge, _DATA_TYPE)

SampleStrategyFactory.direct_register(SampleByNumSamples, _DATA_TYPE, SampleStrategyType.NumSamples)

SpawnFactory.direct_register(Spawn, _DATA_TYPE)
SplitFactory.direct_register(Split, _DATA_TYPE)
