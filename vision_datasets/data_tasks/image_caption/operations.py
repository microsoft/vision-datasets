from ...common import DatasetTypes
from ...data_manifest import GenerateCocoDictBase, ImageLabelManifest, MergeStrategyType, SampleByNumSamples, SampleFewShot, SampleStrategyType, SingleTaskMergeWithIndepedentImages, Spawn, Split
from ...factory.operations import CocoDictGeneratorFactory, ManifestMergeStrategyFactory, SampleStrategyFactory, SpawnFactory, SplitFactory

_DATA_TYPE = DatasetTypes.IMAGE_CAPTION


@CocoDictGeneratorFactory.register(_DATA_TYPE)
class ImageCaptionCocoDictGenerator(GenerateCocoDictBase):
    def process_labels(self, coco_ann, label: ImageLabelManifest):
        coco_ann['caption'] = label.label_data


ManifestMergeStrategyFactory.direct_register(SingleTaskMergeWithIndepedentImages, _DATA_TYPE, MergeStrategyType.IndependentImages)


SampleStrategyFactory.direct_register(SampleByNumSamples, _DATA_TYPE, SampleStrategyType.NumSamples)
SampleStrategyFactory.direct_register(SampleFewShot, _DATA_TYPE, SampleStrategyType.FewShot)

SpawnFactory.direct_register(Spawn, _DATA_TYPE)

SplitFactory.direct_register(Split, _DATA_TYPE)
