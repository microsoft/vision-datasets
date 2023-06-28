import pytest

from vision_datasets.common import DatasetTypes

from ..resources.util import coco_database
from .coco_adaptor_base import BaseCocoAdaptor


class TestImageMatting(BaseCocoAdaptor):
    TASK = DatasetTypes.IMAGE_MATTING

    @pytest.mark.parametrize("coco_dict", coco_database[TASK])
    def test_create_data_manifest(self, coco_dict):
        super().test_create_data_manifest(coco_dict)

    @pytest.mark.parametrize("coco_dict", coco_database[TASK])
    def test_create_data_manifest_with_additional_info(self, coco_dict):
        super().test_create_data_manifest_with_additional_info(coco_dict)
