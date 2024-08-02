import copy
from vision_datasets.common import DatasetManifest, DatasetManifestWithMultiImageLabel
from ..resources.util import coco_dict_to_manifest


class BaseCocoAdaptor:
    def test_create_data_manifest(self, coco_dict, schema: dict = None):
        manifest = coco_dict_to_manifest(self.TASK, coco_dict, schema)
        self.check(manifest, coco_dict)
        return manifest

    def test_create_data_manifest_with_additional_info(self, coco_dict, schema: dict = None):
        coco_dict = copy.deepcopy(coco_dict)
        for img in coco_dict['images']:
            img['img_field_1'] = 1
            img['img_field_2'] = 2

        for ann in coco_dict['annotations']:
            ann['ann_field_1'] = 1
            ann['ann_field_2'] = 2

        if 'categories' in coco_dict:
            for cat in coco_dict['categories']:
                cat['cat_field_1'] = 1
                cat['cat_field_2'] = 2

        coco_dict['dataset_field_1'] = 1
        coco_dict['dataset_field_2'] = 2

        manifest = coco_dict_to_manifest(self.TASK, coco_dict, schema)
        self.check(manifest, coco_dict)
        for img in manifest.images:
            assert img.additional_info.get('img_field_1') == 1
            assert img.additional_info.get('img_field_2') == 2
            for ann in img.labels:
                assert ann.additional_info.get('ann_field_1') == 1
                assert ann.additional_info.get('ann_field_2') == 2

        if isinstance(manifest, DatasetManifestWithMultiImageLabel):
            for ann in manifest.annotations:
                assert ann.additional_info.get('ann_field_1') == 1
                assert ann.additional_info.get('ann_field_2') == 2

        if 'categories' in coco_dict:
            for cat in manifest.categories:
                assert cat.additional_info.get('cat_field_1') == 1
                assert cat.additional_info.get('cat_field_2') == 2

        assert manifest.additional_info.get('dataset_field_1') == 1
        assert manifest.additional_info.get('dataset_field_2') == 2

    def check(self, manifest, coco_dict):
        assert len(manifest.images) == len(coco_dict['images'])
        categories = coco_dict.get('categories')
        if categories:
            assert manifest.categories and len(manifest.categories) == len(categories)
        if isinstance(manifest, DatasetManifest):
            assert sum([len(img.labels) for img in manifest.images]) == len(coco_dict['annotations'])
        elif isinstance(manifest, DatasetManifestWithMultiImageLabel):
            assert manifest.categories is None
            assert len(manifest.annotations) == len(coco_dict['annotations'])
            img_id_set = set(range(len(manifest.images)))
            img_id_coco_to_manifest = {im['id']: id for id, im in enumerate(coco_dict['images'])}
            for id, ann in enumerate(manifest.annotations):
                assert all([img_id in img_id_set for img_id in ann.img_ids])
                assert ann.id == coco_dict['annotations'][id]['id']
                coco_img_ids = coco_dict['annotations'][id]['image_ids']
                assert ann.img_ids == [img_id_coco_to_manifest[coco_img_id] for coco_img_id in coco_img_ids]
        else:
            raise ValueError(f"Unknown manifest type: {type(manifest)}")
