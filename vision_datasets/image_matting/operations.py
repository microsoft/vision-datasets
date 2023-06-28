from ..common import DatasetTypes, GenerateCocoDictBase, ImageLabelManifest, SampleByNumSamples, SampleFewShot, SampleStrategyType, SingleTaskMerge, Spawn, Split, CocoDictGeneratorFactory, \
    ManifestMergeStrategyFactory, SampleStrategyFactory, SpawnFactory, SplitFactory

_DATA_TYPE = DatasetTypes.IMAGE_MATTING


@CocoDictGeneratorFactory.register(_DATA_TYPE)
class ImageMattingCocoDictGenerator(GenerateCocoDictBase):
    def process_labels(self, coco_ann, label: ImageLabelManifest):
        coco_ann['label'] = label.label_path


ManifestMergeStrategyFactory.direct_register(SingleTaskMerge, _DATA_TYPE)


SampleStrategyFactory.direct_register(SampleByNumSamples, _DATA_TYPE, SampleStrategyType.NumSamples)
SampleStrategyFactory.direct_register(SampleFewShot, _DATA_TYPE, SampleStrategyType.FewShot)

SpawnFactory.direct_register(Spawn, _DATA_TYPE)
SplitFactory.direct_register(Split, _DATA_TYPE)
