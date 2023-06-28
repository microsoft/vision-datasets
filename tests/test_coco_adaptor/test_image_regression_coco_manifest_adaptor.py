import pytest

from vision_datasets.common import DatasetTypes

from ..resources.util import coco_database, coco_dict_to_manifest
from .coco_adaptor_base import BaseCocoAdaptor


class TestImageRegression(BaseCocoAdaptor):
    TASK = DatasetTypes.IMAGE_REGRESSION

    @pytest.mark.parametrize("coco_dict", coco_database[TASK])
    def test_create_data_manifest(self, coco_dict):
        super().test_create_data_manifest(coco_dict)

    @pytest.mark.parametrize("coco_dict", coco_database[TASK])
    def test_create_data_manifest_with_additional_info(self, coco_dict):
        super().test_create_data_manifest_with_additional_info(coco_dict)

    def test_create_data_manifest_when_multiple_annotation_per_image_should_fail(self):
        coco_dict = {
            "images": [{"id": 1, "file_name": "train_images.zip@1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "target": 1.0},
                {"id": 2, "image_id": 1, "target": 3.0},
                {"id": 3, "image_id": 2, "target": 2.0},
            ]
        }

        with pytest.raises(ValueError, match='image with id 1 has unexpected number of annotations 1 for DatasetTypes.IMAGE_REGRESSION dataset.'):
            coco_dict_to_manifest(self.TASK, coco_dict)
