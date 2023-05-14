import pytest
from vision_datasets import DatasetTypes
from .coco_adaptor_base import BaseCocoAdaptor
from ..resources.util import coco_database


class TestImageMatting(BaseCocoAdaptor):
    TASK = DatasetTypes.IMAGE_MATTING

    @pytest.mark.parametrize("coco_dict", coco_database[TASK])
    def test_create_data_manifest(self, coco_dict):
        super().test_create_data_manifest(coco_dict)
