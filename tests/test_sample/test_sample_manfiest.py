import itertools

import pytest

from vision_datasets.common import DatasetTypes, ManifestSampler, SampleByFewShotConfig, SampleByNumSamplesConfig, SampleStrategyFactory, SampleStrategyType

from ..resources.util import TYPES_WITH_CATEGORIES, coco_database, coco_dict_to_manifest


class TestSampleManifestNumSamples:
    @pytest.mark.parametrize("task, coco_dict, with_replacement",
                             [x[0] + (x[1],)
                              for x in itertools.product(
                                  [(task, coco_dict) for task, coco_dicts in coco_database.items() if task != DatasetTypes.MULTITASK for coco_dict in coco_dicts],
                                  [True, False])])
    def test_sample_data_manifest_by_n_samples_single_task(self, task, coco_dict, with_replacement):
        manifest = coco_dict_to_manifest(task, coco_dict)
        n_samples = max(len(manifest.images) // 2, 1)
        sampler_strategy = SampleStrategyFactory.create(task, SampleStrategyType.NumSamples, SampleByNumSamplesConfig(0, with_replacement, n_samples))
        sampler = ManifestSampler(sampler_strategy)
        sampled_manifest = sampler.run(manifest)
        assert n_samples == len(sampled_manifest.images)


@pytest.mark.parametrize("task, coco_dict",
                         [(task, coco_dict) for task, coco_dicts in coco_database.items() if task in TYPES_WITH_CATEGORIES for coco_dict in coco_dicts])
class TestSampleManifestFewShots:
    def test_sample_data_manifest_by_few_shot_single_task(self, task, coco_dict):
        manifest = coco_dict_to_manifest(task, coco_dict)
        n_few_shot = 1
        sampler_strategy = SampleStrategyFactory.create(task, SampleStrategyType.FewShot, SampleByFewShotConfig(0, n_few_shot))
        sampler = ManifestSampler(sampler_strategy)
        sampler.run(manifest)

    def test_sample_data_manifest_by_few_shot_single_task_should_throw(self, task, coco_dict):
        manifest = coco_dict_to_manifest(task, coco_dict)
        n_few_shot = len(manifest.images) + 1
        sampler_strategy = SampleStrategyFactory.create(task, SampleStrategyType.FewShot, SampleByFewShotConfig(0, n_few_shot))
        sampler = ManifestSampler(sampler_strategy)
        with pytest.raises(RuntimeError, match=fr"Couldn't find {n_few_shot} samples for some classes:.*"):
            sampler.run(manifest)
