from ..constants import AnnotationFormats, DatasetTypes, Usages
from ..data_manifest.iris_data_manifest_adaptor import IrisManifestAdaptor
from ..dataset_info import BaseDatasetInfo
from ..factory import CocoManifestAdaptorFactory
from ..utils import construct_full_url_or_path_func


class DataManifestFactory:
    @staticmethod
    def create(dataset_info: BaseDatasetInfo, usage: Usages, container_sas_or_root_dir: str = None):
        if dataset_info.data_format == AnnotationFormats.IRIS:
            return IrisManifestAdaptor.create_dataset_manifest(dataset_info, usage, container_sas_or_root_dir)

        if dataset_info.data_format == AnnotationFormats.COCO:
            container_sas_or_root_dir = construct_full_url_or_path_func(container_sas_or_root_dir, dataset_info.root_folder)('')
            if dataset_info.type == DatasetTypes.MULTITASK:
                coco_file_by_task = {k: sub_taskinfo.index_files.get(usage) for k, sub_taskinfo in dataset_info.sub_task_infos.items()}
                data_type_by_task = {k: sub_taskinfo.type for k, sub_taskinfo in dataset_info.sub_task_infos.items()}
                adaptor = CocoManifestAdaptorFactory.create(DatasetTypes.MULTITASK, data_type_by_task)
                return adaptor.create_dataset_manifest(coco_file_by_task, container_sas_or_root_dir)

            adaptor = CocoManifestAdaptorFactory.create(dataset_info.type)
            return adaptor.create_dataset_manifest(dataset_info.index_files.get(usage), container_sas_or_root_dir)
