from .constants import AnnotationFormats, DatasetTypes, Usages


def _data_type_to_enum(val: str):
    # mapping from legacy names before pkg version 1.0.0

    legacy_mapping = {
        'classification_multilabel': DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL,
        'classification_multiclass': DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS,
        'object_detection': DatasetTypes.IMAGE_OBJECT_DETECTION,
        'image_retrieval': DatasetTypes.TEXT_2_IMAGE_RETRIEVAL
    }

    if val.lower() in legacy_mapping:
        return legacy_mapping[val.lower()]

    return DatasetTypes[val.upper()]


class DatasetInfoFactory:
    @staticmethod
    def create(dataset_info_dict: dict):
        data_type = _data_type_to_enum(dataset_info_dict.get('type'))
        if data_type == DatasetTypes.MULTITASK:
            return MultiTaskDatasetInfo(dataset_info_dict)
        return DatasetInfo(dataset_info_dict)


class BaseDatasetInfo:
    """
    Info fields common to both all datasets regardless of whether it is coco or iris, single task or multitask
    """

    def __init__(self, dataset_info_dict):
        self.name = dataset_info_dict['name']
        self.version = dataset_info_dict.get('version', 1)
        self.type = _data_type_to_enum(dataset_info_dict['type'])
        self.root_folder = dataset_info_dict.get('root_folder')
        self.description = dataset_info_dict.get('description', '')
        self.data_format = AnnotationFormats[dataset_info_dict.get('format', 'IRIS').upper()]


class DatasetInfo(BaseDatasetInfo):
    def __init__(self, dataset_info_dict):
        data_type = _data_type_to_enum(dataset_info_dict.get('type'))
        if data_type == DatasetTypes.MULTITASK:
            raise ValueError
        super(DatasetInfo, self).__init__(dataset_info_dict)

        self.index_files = dict()
        self.files_for_local_usage = dict()
        for usage in Usages:
            usage_str = usage.name.lower()
            if usage_str in dataset_info_dict:
                self.index_files[usage] = dataset_info_dict[usage_str]['index_path']
                self.files_for_local_usage[usage] = dataset_info_dict[usage_str].get('files_for_local_usage', [])

        if self.data_format == AnnotationFormats.IRIS:
            self.labelmap = dataset_info_dict.get('labelmap')
            self.image_metadata_path = dataset_info_dict.get('image_metadata_path')
        else:
            self.labelmap = None
            self.image_metadata_path = None


class MultiTaskDatasetInfo(BaseDatasetInfo):
    def __init__(self, dataset_info_dict):
        if 'tasks' not in dataset_info_dict:
            raise ValueError
        data_type = _data_type_to_enum(dataset_info_dict.get('type'))
        if data_type != DatasetTypes.MULTITASK:
            raise ValueError

        super(MultiTaskDatasetInfo, self).__init__(dataset_info_dict)

        tasks = dataset_info_dict['tasks']
        info_dict = {}
        for task_name, task_info in tasks.items():
            info_dict[task_name] = DatasetInfo({**dataset_info_dict, **task_info})

        self.sub_task_infos = info_dict

    @property
    def task_names(self):
        return list(self.sub_task_infos.keys())
