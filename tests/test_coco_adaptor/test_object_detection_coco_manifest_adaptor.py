import pytest
import copy
from vision_datasets.common import DatasetTypes
from .coco_adaptor_base import BaseCocoAdaptor
from ..resources.util import coco_database


class TestObjectDetection(BaseCocoAdaptor):
    TASK = DatasetTypes.IMAGE_OBJECT_DETECTION

    @pytest.mark.parametrize("coco_dict", coco_database[TASK])
    def test_create_data_manifest(self, coco_dict):
        super().test_create_data_manifest(coco_dict)

    @pytest.mark.parametrize("coco_dict", coco_database[TASK])
    def test_create_data_manifest_with_additional_info(self, coco_dict):
        super().test_create_data_manifest_with_additional_info(coco_dict)

    def test_iscrowd_working(self):
        coco_dict = copy.deepcopy(coco_database[self.TASK][0])
        coco_dict['annotations'][0]['iscrowd'] = 1
        super().test_create_data_manifest(coco_dict)

    def check(self, manifest, coco_dict):
        super().check(manifest, coco_dict)
        is_crowd_cnt_in_manifest = sum([sum([0 if x.additional_info.get('iscrowd', 0) == 0 else 1 for x in image.labels]) for image in manifest.images])
        is_crowd_cnt_in_coco = sum([0 if ann.get('iscrowd', 0) == 0 else 1 for ann in coco_dict['annotations']])
        assert is_crowd_cnt_in_coco == is_crowd_cnt_in_manifest
