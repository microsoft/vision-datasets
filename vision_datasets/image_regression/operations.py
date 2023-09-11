from ..common import DatasetTypes, GenerateCocoDictBase, ImageLabelManifest, SampleByNumSamples, SampleStrategyType, SingleTaskMerge, Spawn, Split, CocoDictGeneratorFactory, \
    ManifestMergeStrategyFactory, SampleStrategyFactory, SpawnFactory, SplitFactory, StandAloneImageListGeneratorFactory, GenerateStandAloneImageListBase, \
    ImageDataManifest, DatasetManifest
from .manifest import ImageRegressionLabelManifest

_DATA_TYPE = DatasetTypes.IMAGE_REGRESSION


@CocoDictGeneratorFactory.register(_DATA_TYPE)
class ImageRegressionCocoDictGenerator(GenerateCocoDictBase):
    def process_labels(self, coco_ann, label: ImageLabelManifest):
        coco_ann['target'] = label.label_data


ManifestMergeStrategyFactory.direct_register(SingleTaskMerge, _DATA_TYPE)

SampleStrategyFactory.direct_register(SampleByNumSamples, _DATA_TYPE, SampleStrategyType.NumSamples)

SpawnFactory.direct_register(Spawn, _DATA_TYPE)
SplitFactory.direct_register(Split, _DATA_TYPE)


@StandAloneImageListGeneratorFactory.register(_DATA_TYPE)
class ImageRegressionStandAloneImageListGenerator(GenerateStandAloneImageListBase):
    def _generate_label(self, label: ImageRegressionLabelManifest, image: ImageDataManifest, manifest: DatasetManifest) -> dict:
        return {'target': label.target}
