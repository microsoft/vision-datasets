import pytest
import pickle
from vision_datasets.common import DatasetTypes
from ..resources.util import coco_dict_to_manifest, coco_database, schema_database


class TestManifestIsPickleable:
    @pytest.mark.parametrize("task, coco_dict", [(task, coco_dict) for task, coco_dicts in coco_database.items() if task != DatasetTypes.KEY_VALUE_PAIR for coco_dict in coco_dicts])
    def test_create_data_manifest(self, task, coco_dict):
        manifest = coco_dict_to_manifest(task, coco_dict)
        self._check_pickleable(manifest)

    @pytest.mark.parametrize("coco_dict, schema", zip(coco_database[DatasetTypes.KEY_VALUE_PAIR], schema_database[DatasetTypes.KEY_VALUE_PAIR]))
    def test_create_key_value_pair_manifest(self, coco_dict, schema):
        manifest = coco_dict_to_manifest(DatasetTypes.KEY_VALUE_PAIR, coco_dict, schema)
        self._check_pickleable(manifest)

    @staticmethod
    def _check_pickleable(manifest):
        serialized = pickle.dumps(manifest)
        deserialized = pickle.loads(serialized)
        assert manifest == deserialized
