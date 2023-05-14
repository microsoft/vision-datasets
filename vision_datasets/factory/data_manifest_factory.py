from ..common import AnnotationFormats, DatasetTypes
from ..data_manifest.iris_data_manifest_adaptor import IrisManifestAdaptor
from ..factory import CocoManifestAdaptorFactory


class DataManifestFactory:
    @staticmethod
    def create(dataset_info, usage: str, container_sas_or_root_dir: str = None):
        annotation_format = AnnotationFormats[dataset_info.data_format]

        if annotation_format == AnnotationFormats.IRIS:
            return IrisManifestAdaptor.create_dataset_manifest(dataset_info, usage, container_sas_or_root_dir)

        if annotation_format == AnnotationFormats.COCO:
            from ..common.utils import construct_full_url_or_path_generator
            container_sas_or_root_dir = construct_full_url_or_path_generator(container_sas_or_root_dir, dataset_info.root_folder)('')
            if dataset_info.type == DatasetTypes.MULTITASK:
                coco_file_by_task = {k: sub_taskinfo.index_files.get(usage) for k, sub_taskinfo in dataset_info.sub_task_infos.items()}
                data_type_by_task = {k: sub_taskinfo.type for k, sub_taskinfo in dataset_info.sub_task_infos.items()}
                adaptor = CocoManifestAdaptorFactory.create(DatasetTypes.MULTITASK, data_type_by_task)
                return adaptor.create_dataset_manifest(coco_file_by_task, container_sas_or_root_dir)

            adaptor = CocoManifestAdaptorFactory.create(dataset_info.type, data_type_by_task)
            return adaptor.create_dataset_manifest(dataset_info.index_files.get(usage), container_sas_or_root_dir)
