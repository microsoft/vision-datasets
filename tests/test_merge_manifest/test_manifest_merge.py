import copy

import pytest

from vision_datasets.common import DatasetTypes, ManifestMerger, ManifestMergeStrategyFactory

from ..resources.util import TYPES_WITH_CATEGORIES, coco_database, coco_dict_to_manifest

DATA_TYPES = [x for x in ManifestMergeStrategyFactory.list_data_types() if x != DatasetTypes.MULTITASK]


class TestMergeManifest:
    @pytest.mark.parametrize("data_type, coco_dicts", [(data_type, coco_database[data_type]) for data_type in DATA_TYPES if data_type not in TYPES_WITH_CATEGORIES])
    def test_merge_data_manifest_single_task_without_categories(self, data_type, coco_dicts):
        coco_dict_1, coco_dict_2 = coco_dicts[0], coco_dicts[-1]
        manifest1, manifest2, merged = self.merge_data_manifest_single_task(data_type, coco_dict_1, coco_dict_2)
        self.check(manifest1, manifest2, merged)

    @pytest.mark.parametrize("data_type, coco_dicts", [(data_type, coco_database[data_type]) for data_type in TYPES_WITH_CATEGORIES])
    def test_merge_data_manifest_single_task_with_same_categories(self, data_type, coco_dicts):
        coco_dict_1, coco_dict_2 = coco_dicts[0], coco_dicts[0]
        manifest1, manifest2, merged = self.merge_data_manifest_single_task(data_type, coco_dict_1, coco_dict_2)
        self.check(manifest1, manifest2, merged)

    @pytest.mark.parametrize("data_type, coco_dicts", [(data_type, coco_database[data_type]) for data_type in TYPES_WITH_CATEGORIES])
    def test_merge_data_manifest_single_task_with_different_categories(self, data_type, coco_dicts):
        coco_dict_1 = coco_dicts[0]
        coco_dict_2 = copy.deepcopy(coco_dict_1)
        for x in coco_dict_2['categories']:
            x['name'] += '_unique'

        manifest1, manifest2, merged = self.merge_data_manifest_single_task(data_type, coco_dict_1, coco_dict_2)
        self.check(manifest1, manifest2, merged)

    @pytest.mark.parametrize("data_type, coco_dicts", [(data_type, coco_database[data_type]) for data_type in TYPES_WITH_CATEGORIES])
    def test_merge_data_manifest_single_task_with_overlapping_categories(self, data_type, coco_dicts):
        coco_dict_1 = [coco_dict for coco_dict in coco_dicts if len(coco_dict['categories']) > 1][0]
        coco_dict_2 = copy.deepcopy(coco_dict_1)
        coco_dict_2['categories'][0]['name'] += '_unique'
        manifest1, manifest2, merged = self.merge_data_manifest_single_task(data_type, coco_dict_1, coco_dict_2)
        self.check(manifest1, manifest2, merged)

    def check(self, manifest1, manifest2, merged):
        assert len(manifest1.images) + len(manifest2.images) == len(merged.images)
        n_categories = len(set([x.name for x in (manifest1.categories or []) + (manifest2.categories or [])]))
        assert n_categories == len(merged.categories or [])

    @staticmethod
    def merge_data_manifest_single_task(data_type, coco_dict_1, coco_dict_2):
        manifest1 = coco_dict_to_manifest(data_type, coco_dict_1)
        manifest2 = coco_dict_to_manifest(data_type, coco_dict_2)
        strategy = ManifestMergeStrategyFactory.create(data_type)
        merger = ManifestMerger(strategy)
        merged = merger.run(manifest1, manifest2)
        return manifest1, manifest2, merged
