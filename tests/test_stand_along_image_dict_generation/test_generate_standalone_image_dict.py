import pytest
from unittest.mock import patch

from vision_datasets.common import StandAloneImageListGeneratorFactory, Base64Utils, DatasetTypes

from ..resources.util import coco_database, coco_dict_to_manifest

DATASET_TYPES = StandAloneImageListGeneratorFactory.list_data_types()


class TestManifestToStandAloneImageDict:

    @pytest.mark.parametrize("coco_dict, task", [(coco_dict, task) for task in DATASET_TYPES for coco_dict in coco_database[task]])
    def test_manifest_to_standalone_image_dict_flattened(self, coco_dict, task):
        self._test_manifest_to_standalone_image_dict_flattened(coco_dict, task, True, len(coco_dict['annotations']))

    @pytest.mark.parametrize("coco_dict, task", [(coco_dict, task) for task in DATASET_TYPES for coco_dict in coco_database[task]])
    def test_manifest_to_standalone_image_dict_not_flattened(self, coco_dict, task):
        def custom_item_check(item): return isinstance(item['labels'], list)
        self._test_manifest_to_standalone_image_dict_flattened(coco_dict, task, False, len(coco_dict['images']), custom_item_check)

    def _test_manifest_to_standalone_image_dict_flattened(self, coco_dict, task, flatten, expected_num_items, custom_check=lambda _: True):
        manifest = coco_dict_to_manifest(task, coco_dict)
        with patch.object(Base64Utils, 'file_to_b64_str', return_value="b64string") as mocked_method:
            coco_generator = StandAloneImageListGeneratorFactory.create(task, flatten)
            n_items = 0
            for item in coco_generator.run(manifest):
                assert item['image'] == "b64string"
                assert custom_check(item)
                n_items += 1
            assert n_items == expected_num_items
            self._mock_check_by_type(task, mocked_method, coco_dict)

    def _mock_check_by_type(self, data_type, mocked_method, coco_dict: dict):
        if data_type == DatasetTypes.IMAGE_MATTING:
            assert mocked_method.call_count == len(coco_dict['images']) + len(coco_dict['annotations'])
        else:
            assert mocked_method.call_count == len(coco_dict['images'])
