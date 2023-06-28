import pytest

from vision_datasets.common import DatasetTypes, Spawn, SpawnConfig

from ..resources.util import coco_database, coco_dict_to_manifest


class TestSpawn:
    @pytest.mark.parametrize("task, coco_dict", [(task, coco_dict) for task, coco_dicts in coco_database.items() if task != DatasetTypes.MULTITASK for coco_dict in coco_dicts])
    def test_spawn_single_task(self, task, coco_dict):
        manifest = coco_dict_to_manifest(task, coco_dict)
        n_target = len(manifest.images) * 2
        cfg = SpawnConfig(0, n_target)
        sp = Spawn(cfg)
        spawned_manifest = sp.run(manifest)
        assert len(spawned_manifest.images) == n_target
