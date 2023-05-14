from ...common import DatasetTypes
from ...data_manifest import (BalancedInstanceWeightsGenerator, GenerateCocoDictBase, ImageLabelManifest, MergeStrategyType, SampleByNumSamples, SampleFewShot, SampleStrategyType,
                              SingleTaskMergeWithIndepedentImages, Spawn, SplitWithCategories)
from ...factory.operations import BalancedInstanceWeightsFactory, CocoDictGeneratorFactory, ManifestMergeStrategyFactory, SampleStrategyFactory, SpawnFactory, SplitFactory


@CocoDictGeneratorFactory.register(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)
@CocoDictGeneratorFactory.register(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)
class ImageClassificationCocoDictGenerator(GenerateCocoDictBase):
    def process_labels(self, coco_ann, label: ImageLabelManifest):
        coco_ann['category_id'] = label.label_data + 1


ManifestMergeStrategyFactory.direct_register(SingleTaskMergeWithIndepedentImages, DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, MergeStrategyType.IndependentImages)
ManifestMergeStrategyFactory.direct_register(SingleTaskMergeWithIndepedentImages, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, MergeStrategyType.IndependentImages)


SampleStrategyFactory.direct_register(SampleByNumSamples, DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, SampleStrategyType.NumSamples)
SampleStrategyFactory.direct_register(SampleFewShot, DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, SampleStrategyType.FewShot)

SampleStrategyFactory.direct_register(SampleByNumSamples, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, SampleStrategyType.NumSamples)
SampleStrategyFactory.direct_register(SampleFewShot, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, SampleStrategyType.FewShot)

SpawnFactory.direct_register(Spawn, DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)
SpawnFactory.direct_register(Spawn, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)


BalancedInstanceWeightsFactory.direct_register(BalancedInstanceWeightsGenerator, DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)
BalancedInstanceWeightsFactory.direct_register(BalancedInstanceWeightsGenerator, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)

SplitFactory.direct_register(SplitWithCategories, DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)
SplitFactory.direct_register(SplitWithCategories, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)
