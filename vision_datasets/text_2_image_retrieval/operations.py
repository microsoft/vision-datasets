from ..common import CocoDictGeneratorFactory, DatasetTypes, GenerateCocoDictBase, ImageLabelManifest, ManifestMergeStrategyFactory, SampleByNumSamples, SampleStrategyFactory, \
    SampleStrategyType, SingleTaskMerge, Spawn, SpawnFactory, Split, SplitFactory, StandAloneImageListGeneratorFactory, GenerateStandAloneImageListBase, \
    DatasetManifest, ImageDataManifest
from .manifest import Text2ImageRetrievalLabelManifest

_DATA_TYPE = DatasetTypes.TEXT_2_IMAGE_RETRIEVAL


@CocoDictGeneratorFactory.register(_DATA_TYPE)
class Text2ImageRetrievalCocoDictGenerator(GenerateCocoDictBase):
    def process_labels(self, coco_ann, label: ImageLabelManifest):
        coco_ann['query'] = label.label_data


ManifestMergeStrategyFactory.direct_register(SingleTaskMerge, _DATA_TYPE)


SampleStrategyFactory.direct_register(SampleByNumSamples, _DATA_TYPE, SampleStrategyType.NumSamples)

SpawnFactory.direct_register(Spawn, _DATA_TYPE)
SplitFactory.direct_register(Split, _DATA_TYPE)


@StandAloneImageListGeneratorFactory.register(_DATA_TYPE)
class Text2ImageRetrievalStandAloneImageListGenerator(GenerateStandAloneImageListBase):
    def _generate_label(self, label: Text2ImageRetrievalLabelManifest, image: ImageDataManifest, manifest: DatasetManifest) -> dict:
        return {'query': label.query}
