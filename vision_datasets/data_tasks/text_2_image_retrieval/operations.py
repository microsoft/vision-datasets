from ...common import DatasetTypes
from ...data_manifest import GenerateCocoDictBase, ImageLabelManifest, SampleByNumSamples, SampleFewShot, SampleStrategyType, SingleTaskMerge, Spawn, Split
from ...factory.operations import CocoDictGeneratorFactory, ManifestMergeStrategyFactory, SampleStrategyFactory, SpawnFactory, SplitFactory

_DATA_TYPE = DatasetTypes.TEXT_2_IMAGE_RETRIEVAL


@CocoDictGeneratorFactory.register(_DATA_TYPE)
class Text2ImageRetrievalCocoDictGenerator(GenerateCocoDictBase):
    def process_labels(self, coco_ann, label: ImageLabelManifest):
        coco_ann['query'] = label.label_data


ManifestMergeStrategyFactory.direct_register(SingleTaskMerge, _DATA_TYPE)


SampleStrategyFactory.direct_register(SampleByNumSamples, _DATA_TYPE, SampleStrategyType.NumSamples)
SampleStrategyFactory.direct_register(SampleFewShot, _DATA_TYPE, SampleStrategyType.FewShot)

SpawnFactory.direct_register(Spawn, _DATA_TYPE)
SplitFactory.direct_register(Split, _DATA_TYPE)
