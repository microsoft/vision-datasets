import copy
import numpy as np

from ..common import DatasetTypes, \
    MultiImageCocoDictGenerator, \
    MultiImageDatasetSingleTaskMerge, CocoDictGeneratorFactory, ManifestMergeStrategyFactory, \
    SampleStrategyFactory, SampleStrategyType, SampleStrategy
from .manifest import KeyValuePairLabelManifest, KeyValuePairDatasetManifest

_DATA_TYPE = DatasetTypes.KEY_VALUE_PAIR


@CocoDictGeneratorFactory.register(_DATA_TYPE)
class KeyValuePairCocoDictGenerator(MultiImageCocoDictGenerator):
    def process_labels(self, coco_ann, label: KeyValuePairLabelManifest):
        coco_ann[KeyValuePairLabelManifest.LABEL_KEY] = label.fields
        if label.text is not None:
            coco_ann[KeyValuePairLabelManifest.TEXT_INPUT_KEY] = label.text

    def _generate_images(self, manifest: KeyValuePairDatasetManifest):
        images = super()._generate_images(manifest)
        # add metadata field if exists
        for img, img_manifest in zip(images, manifest.images):
            if img_manifest.additional_info is not None and 'metadata' in img_manifest.additional_info:
                img['metadata'] = img_manifest.additional_info['metadata']
        return images


@ManifestMergeStrategyFactory.register(_DATA_TYPE)
class KeyValuePairDatasetMerge(MultiImageDatasetSingleTaskMerge):
    def merge(self, *args: KeyValuePairDatasetManifest):
        schema = args[0].schema
        for manifest in args[1:]:
            if manifest.schema != schema:
                raise ValueError('Schema mismatch')
        return super().merge(*args)


@SampleStrategyFactory.register(_DATA_TYPE, SampleStrategyType.NumSamples)
class KeyValuePairDatasetSampleByNumSamples(SampleStrategy):
    def __init__(self, config):
        super().__init__(config)
        if config.n_samples <= 0:
            raise ValueError('n samples must be greater than zero.')

    def sample(self, manifest: KeyValuePairDatasetManifest) -> KeyValuePairDatasetManifest:
        if not self.config.with_replacement and self.config.n_samples > len(manifest.annotations):
            raise ValueError('When with_replacement is disabled, n_samples must be less than or equal to the number of annotations in the dataset.')

        rng = np.random.default_rng(self.config.random_seed)
        normalized_weights = [w / sum(self.config.weights) for w in self.config.weights] if self.config.weights else None
        sampled_indices = rng.choice(len(manifest.annotations), size=self.config.n_samples, replace=self.config.with_replacement, p=normalized_weights)
        sampled_annotations = [manifest.annotations[i] for i in sampled_indices]

        return KeyValuePairDatasetManifest(copy.deepcopy(manifest.images),
                                           copy.deepcopy(sampled_annotations),
                                           copy.deepcopy(manifest.schema),
                                           copy.deepcopy(manifest.additional_info))
