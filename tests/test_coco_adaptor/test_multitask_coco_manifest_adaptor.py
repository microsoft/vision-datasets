import pytest
from vision_datasets.common import DatasetTypes
from ..resources.util import coco_database, coco_dict_to_manifest_multitask


class TestMultiTask:
    TASK = DatasetTypes.MULTITASK

    @pytest.mark.parametrize("tasks, coco_dicts", coco_database[TASK])
    def test_create_data_manifest(self, tasks, coco_dicts):
        coco_dict_to_manifest_multitask(tasks, coco_dicts)

        # TODO: need to implement more checks
