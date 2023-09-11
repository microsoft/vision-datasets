from ..common import DatasetTypes, GenerateCocoDictBase, ImageLabelManifest, SampleByNumSamples, SampleStrategyType, SingleTaskMerge, Spawn, Split, CocoDictGeneratorFactory, \
    ManifestMergeStrategyFactory, SampleStrategyFactory, SpawnFactory, SplitFactory, StandAloneImageListGeneratorFactory, GenerateStandAloneImageListBase, \
    ImageDataManifest, DatasetManifest
from .manifest import ImageCaptionLabelManifest

_DATA_TYPE = DatasetTypes.IMAGE_CAPTION


@CocoDictGeneratorFactory.register(_DATA_TYPE)
class ImageCaptionCocoDictGenerator(GenerateCocoDictBase):
    def process_labels(self, coco_ann, label: ImageLabelManifest):
        coco_ann['caption'] = label.label_data


ManifestMergeStrategyFactory.direct_register(SingleTaskMerge, _DATA_TYPE)


SampleStrategyFactory.direct_register(SampleByNumSamples, _DATA_TYPE, SampleStrategyType.NumSamples)

SpawnFactory.direct_register(Spawn, _DATA_TYPE)

SplitFactory.direct_register(Split, _DATA_TYPE)

StandAloneImageListGeneratorFactory.direct_register(GenerateStandAloneImageListBase, _DATA_TYPE)


@StandAloneImageListGeneratorFactory.register(_DATA_TYPE)
class ImageCaptionStandAloneImageListGenerator(GenerateStandAloneImageListBase):
    def _generate_label(self, label: ImageCaptionLabelManifest, image: ImageDataManifest, manifest: DatasetManifest):
        return {'caption': label.caption}
