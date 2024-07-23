from ..common import DatasetTypes, MultiImageCocoDictGenerator, \
    MultiImageDatasetSingleTaskMerge, CocoDictGeneratorFactory, ManifestMergeStrategyFactory
    
from .manifest import KVPairLabelManifest, KVPairDatasetManifest

_DATA_TYPE = DatasetTypes.KEY_VALUE_PAIR


@CocoDictGeneratorFactory.register(_DATA_TYPE)
class KVPairCocoDictGenerator(MultiImageCocoDictGenerator):
    def process_labels(self, coco_ann, label: KVPairLabelManifest):
        coco_ann[KVPairLabelManifest.LABEL_KEY] = label.key_value_pairs
        if label.text_input is not None:
            coco_ann[KVPairLabelManifest.INPUT_KEY] = label.text_input

    def _generate_images(self, manifest: KVPairDatasetManifest):
        images = super()._generate_images(manifest)
        # add metadata field if exists
        for img, img_manifest in zip(images, manifest.images):
            if img_manifest.additional_info is not None and 'metadata' in img_manifest.additional_info:
                img['metadata'] = img_manifest.additional_info['metadata']
        return images


@ManifestMergeStrategyFactory.register(_DATA_TYPE)
class KVPairDatasetMerge(MultiImageDatasetSingleTaskMerge):
    def merge(self, *args: KVPairDatasetManifest):
        schema = args[0].schema
        for manifest in args[1:]:
            if manifest.schema != schema:
                raise ValueError('Schema mismatch')
        return super().merge(*args)
