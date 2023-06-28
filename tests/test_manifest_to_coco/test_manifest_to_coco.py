import pytest

from vision_datasets.common import CocoDictGeneratorFactory, DatasetTypes

from ..resources.util import coco_database, coco_dict_to_manifest


class TestManifestToCoco:
    @pytest.mark.parametrize("coco_dict, task", [(coco_dict, task) for task, coco_dicts in coco_database.items() if task != DatasetTypes.MULTITASK for coco_dict in coco_dicts])
    def test_manifest_to_coco_dict(self, coco_dict, task):
        manifest = coco_dict_to_manifest(task, coco_dict)
        coco_generator = CocoDictGeneratorFactory.create(task)
        coco_dict = coco_generator.run(manifest)
