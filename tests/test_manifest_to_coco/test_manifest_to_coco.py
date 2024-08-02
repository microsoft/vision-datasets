import pytest
from vision_datasets.common import CocoDictGeneratorFactory, DatasetTypes

from ..resources.util import coco_database, schema_database, coco_dict_to_manifest


class TestManifestToCoco:
    @pytest.mark.parametrize("coco_dict, task", [(coco_dict, task) for task, coco_dicts in coco_database.items() if task not in [DatasetTypes.MULTITASK, DatasetTypes.KEY_VALUE_PAIR]
                                                 for coco_dict in coco_dicts])
    def test_manifest_to_coco_dict(self, coco_dict, task):
        manifest = coco_dict_to_manifest(task, coco_dict)
        coco_generator = CocoDictGeneratorFactory.create(task)
        coco_dict = coco_generator.run(manifest)

    @pytest.mark.parametrize("coco_dict, schema", zip(coco_database[DatasetTypes.KEY_VALUE_PAIR], schema_database[DatasetTypes.KEY_VALUE_PAIR]))
    def test_key_value_pair_manifest_to_coco_dict(self, coco_dict, schema):
        manifest = coco_dict_to_manifest(DatasetTypes.KEY_VALUE_PAIR, coco_dict, schema)
        coco_generator = CocoDictGeneratorFactory.create(DatasetTypes.KEY_VALUE_PAIR)
        coco_dict = coco_generator.run(manifest)
