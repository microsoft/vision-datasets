import copy

from ..common import DatasetTypes, DatasetManifest, MergeStrategy, ManifestMergeStrategyFactory
from ..common.utils import deep_merge

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

        addional_info = deep_merge([x.additional_info for x in args])
        if not addional_info:
            addional_info = None
        return DatasetManifest([y for x in args for y in x.images], args[0].categories, args[0].data_type, addional_info)

    def check(self, *args: DatasetManifest):
        super().check(*args)
        if not all([x.is_multitask for x in args]):
            raise ValueError('all manifests must be multitask.')

        for i in range(len(args)-1):
            m1, m2 = args[i], args[i+1]
            if m1.data_type != m2.data_type:
                raise ValueError
            if m1.categories != m2.categories:
                raise ValueError


# SampleStrategyFactory.direct_register(SampleByNumSamples, _DATA_TYPE, SampleStrategyType.NumSamples)
# SampleStrategyFactory.direct_register(SampleFewShot, _DATA_TYPE, SampleStrategyType.FewShot)

# SplitFactory.direct_register(Split, _DATA_TYPE)
