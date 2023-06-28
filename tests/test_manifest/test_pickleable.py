import pytest
import pickle
from ..resources.util import coco_dict_to_manifest, coco_database


class TestManifestIsPickleable:
    @pytest.mark.parametrize("task, coco_dict", [(task, coco_dict) for task, coco_dicts in coco_database.items() for coco_dict in coco_dicts])
    def test_create_data_manifest(self, task, coco_dict):
        manifest = coco_dict_to_manifest(task, coco_dict)
        self._check_pickleable(manifest)

    @staticmethod
    def _check_pickleable(manifest):
        serialized = pickle.dumps(manifest)
        deserialized = pickle.loads(serialized)
        assert manifest == deserialized
