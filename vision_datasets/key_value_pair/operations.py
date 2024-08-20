from ..common import DatasetTypes, MultiImageCocoDictGenerator, \
    MultiImageDatasetSingleTaskMerge, CocoDictGeneratorFactory, ManifestMergeStrategyFactory

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
