import copy

from ...common import DatasetTypes
from ...data_manifest import DatasetManifest, MergeStrategy
from ...factory.operations import ManifestMergeStrategyFactory

_DATA_TYPE = DatasetTypes.MULTITASK


@ManifestMergeStrategyFactory.register(_DATA_TYPE)
class MultitaskMerge(MergeStrategy):
    def merge(self, *args: DatasetManifest):
        all_categories = {}
        data_tasks = {}
        for manifest in args:
            for task_name, categories in manifest.categories.items():
                all_categories[task_name] = copy.deepcopy(categories)

            for task_name, tasks in manifest.data_type.items():
                data_tasks[task_name] = copy.deepcopy(tasks)

        return DatasetManifest([y for x in args for y in x.images], args[0].categories, args[0].data_type)

    def check(self, *args: DatasetManifest):
        super().check(*args)
        assert all([x.is_multitask for x in args])

        for i in range(len(args)-1):
            m1, m2 = args[i], args[i+1]
            assert m1.data_type == m2.data_type
            assert m1.categories == m2.categories


# SampleStrategyFactory.direct_register(SampleByNumSamples, _DATA_TYPE, SampleStrategyType.NumSamples)
# SampleStrategyFactory.direct_register(SampleFewShot, _DATA_TYPE, SampleStrategyType.FewShot)

# SplitFactory.direct_register(Split, _DATA_TYPE)
