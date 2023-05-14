import copy

from ...common import DatasetTypes
from ...data_manifest import DatasetManifest, MergeStrategy, MergeStrategyType, SampleByNumSamples, SampleFewShot, SampleStrategyType, Split
from ...factory.operations import ManifestMergeStrategyFactory, SampleStrategyFactory, SplitFactory

_DATA_TYPE = DatasetTypes.MULTITASK


@ManifestMergeStrategyFactory.register(_DATA_TYPE, MergeStrategyType.IndependentImages)
class MultitaskIndependentCategoriesMerge(MergeStrategy):
    def merge(self, *args: DatasetManifest):
        all_categories = {}
        data_tasks = {}
        for manifest in args:
            for task_name, categories in manifest.categories.items():
                all_categories[task_name] = copy.deepcopy(categories)

            for task_name, tasks in manifest.data_type.items():
                data_tasks[task_name] = copy.deepcopy(tasks)

        return DatasetManifest([y for x in args for y in x.images], all_categories, data_tasks)

    def check(self, *args: DatasetManifest):
        """Checking all category names are unique

        Raises:
            ValueError: if duplicate category name exists
        """
        super().check(*args)
        assert args[0].is_multitask
        all_categories = {}
        for manifest in args:
            for task_name, task_categories in manifest.categories.items():
                if task_name in all_categories:
                    raise ValueError(f'Failed to merge dataset manifests, as due to task with name {task_name} exists in more than one manifest.')

                all_categories[task_name] = task_categories


SampleStrategyFactory.direct_register(SampleByNumSamples, _DATA_TYPE, SampleStrategyType.NumSamples)
SampleStrategyFactory.direct_register(SampleFewShot, _DATA_TYPE, SampleStrategyType.FewShot)

SplitFactory.direct_register(Split, _DATA_TYPE)
