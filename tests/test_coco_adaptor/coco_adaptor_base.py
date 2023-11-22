import copy
from ..resources.util import coco_dict_to_manifest


class BaseCocoAdaptor:
    def test_create_data_manifest(self, coco_dict):
        manifest = coco_dict_to_manifest(self.TASK, coco_dict)
        self.check(manifest, coco_dict)
        return manifest

    def test_create_data_manifest_with_additional_info(self, coco_dict):
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

        manifest = coco_dict_to_manifest(self.TASK, coco_dict)
        self.check(manifest, coco_dict)
        for img in manifest.images:
            assert img.additional_info.get('img_field_1') == 1
            assert img.additional_info.get('img_field_2') == 2
            for ann in img.labels:
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
        assert sum([len(img.labels) for img in manifest.images]) == len(coco_dict['annotations'])
