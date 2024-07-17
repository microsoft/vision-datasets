from ..common import DatasetTypes, GenerateCocoDictFromAnnotationWiseManifest, \
    AnnotationWiseSingleTaskMerge, CocoDictGeneratorFactory, ManifestMergeStrategyFactory, \
    AnnotationWiseDatasetManifest
from .manifest import KVPairLabelManifest

_DATA_TYPE = DatasetTypes.KV_PAIR


@CocoDictGeneratorFactory.register(_DATA_TYPE)
class KVPairCocoDictGenerator(GenerateCocoDictFromAnnotationWiseManifest):
    def process_labels(self, coco_ann, label: KVPairLabelManifest):
        coco_ann[KVPairLabelManifest.KV_PAIR_KEY] = label.key_value_pairs
        coco_ann[KVPairLabelManifest.INPUT_KEY] = label.text_input

    def _generate_images(self, manifest: AnnotationWiseDatasetManifest):
        images = super()._generate_images(manifest)
        # add metadata field if exists
        for img, img_manifest in zip(images, manifest.images):
            if img_manifest.additional_info is not None and 'metadata' in img_manifest.additional_info:
                img['metadata'] = img_manifest.additional_info['metadata']
        return images


ManifestMergeStrategyFactory.direct_register(AnnotationWiseSingleTaskMerge, _DATA_TYPE)

# TODO: add other operations such as sample, split
