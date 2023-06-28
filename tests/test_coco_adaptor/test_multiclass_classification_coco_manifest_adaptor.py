import copy
import pytest
from vision_datasets.common import DatasetTypes
from .coco_adaptor_base import BaseCocoAdaptor
from ..resources.util import coco_database, coco_dict_to_manifest


class TestMultiClassClassification(BaseCocoAdaptor):
    TASK = DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS

    @pytest.mark.parametrize("coco_dict", coco_database[TASK])
    def test_create_data_manifest(self, coco_dict):
        super().test_create_data_manifest(coco_dict)

    @pytest.mark.parametrize("coco_dict", coco_database[TASK])
    def test_create_data_manifest_with_additional_info(self, coco_dict):
        super().test_create_data_manifest_with_additional_info(coco_dict)

    def test_create_data_manifest_when_multiple_annotation_per_image_should_fail(self):
        coco_dict = {
            "images": [
                    {"id": 1, "width": 224.0, "height": 224.0, "file_name": "train/1.jpg"},
                    {"id": 2, "width": 224.0, "height": 224.0, "file_name": "train/3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1},
                {"id": 2, "category_id": 2, "image_id": 1},
                {"id": 3, "category_id": 2, "image_id": 2}
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        }
        with pytest.raises(ValueError, match='image with id 1 has unexpected number of annotations 1 for DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS dataset.'):
            coco_dict_to_manifest(self.TASK, coco_dict)

    def test_supercategory_working(self):
        super_category = 'animal'
        coco_dict = copy.deepcopy(coco_database[self.TASK][0])
        for category in coco_dict['categories']:
            category['supercategory'] = super_category
        manifest = coco_dict_to_manifest(self.TASK, coco_dict)
        for category in manifest.categories:
            assert category.super_category == super_category
