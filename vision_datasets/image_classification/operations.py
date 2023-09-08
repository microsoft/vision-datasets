from ..common import DatasetTypes, BalancedInstanceWeightsGenerator, GenerateCocoDictBase, SampleByNumSamples, SampleFewShot, SampleStrategyType, SingleTaskMerge, Spawn, \
    SplitWithCategories, BalancedInstanceWeightsFactory, CocoDictGeneratorFactory, ManifestMergeStrategyFactory, SampleStrategyFactory, SpawnFactory, SplitFactory, \
    StandAloneImageListGeneratorFactory, GenerateStandAloneImageListBase, DatasetManifest, ImageDataManifest
from .manifest import ImageClassificationLabelManifest


@CocoDictGeneratorFactory.register(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)
@CocoDictGeneratorFactory.register(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)
class ImageClassificationCocoDictGenerator(GenerateCocoDictBase):
    def process_labels(self, coco_ann, label: ImageClassificationLabelManifest):
        coco_ann['category_id'] = label.label_data + 1


ManifestMergeStrategyFactory.direct_register(SingleTaskMerge, DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)
ManifestMergeStrategyFactory.direct_register(SingleTaskMerge, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)


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


@StandAloneImageListGeneratorFactory.register(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)
@StandAloneImageListGeneratorFactory.register(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)
class ImageClassificationStandAloneImageListGenerator(GenerateStandAloneImageListBase):
    def _generate_label(self, label: ImageClassificationLabelManifest, image: ImageDataManifest, manifest: DatasetManifest):
        return {'category_name': manifest.categories[label.category_id]}
