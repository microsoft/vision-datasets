import pytest
from vision_datasets.common import DatasetTypes, Spawn, SpawnConfig, SpawnFactory

from ..resources.util import coco_database, coco_dict_to_manifest

DATA_TYPES = [x for x in SpawnFactory.list_data_types() if x != DatasetTypes.MULTITASK]


class TestSpawn:
    @pytest.mark.parametrize("task, coco_dict", [(task, coco_dict) for task in DATA_TYPES for coco_dict in coco_database[task]])
    def test_spawn_single_task(self, task, coco_dict):
        manifest = coco_dict_to_manifest(task, coco_dict)
        n_target = len(manifest.images) * 2
        cfg = SpawnConfig(0, n_target)
        sp = Spawn(cfg)
        spawned_manifest = sp.run(manifest)
        assert len(spawned_manifest.images) == n_target
