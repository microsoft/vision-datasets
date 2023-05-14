import pathlib
import typing

from ...common import DatasetTypes
from ...data_manifest.coco_manifest_adaptor import CocoManifestAdaptorBase
from ...data_manifest.utils import generate_multitask_dataset_manifest
from ...factory import CocoManifestAdaptorFactory


@CocoManifestAdaptorFactory.register(DatasetTypes.MULTITASK)
class MultiTaskCocoManifestAdaptor(CocoManifestAdaptorBase):
    def __init__(self, data_tasks: dict) -> None:
        super().__init__(data_tasks)

    def create_dataset_manifest(self, coco_file_path_or_url: typing.Union[str, dict, pathlib.Path], container_sas_or_root_dir: str = None):
        """ construct a dataset manifest out of coco file
        Args:
            coco_file_path_or_url (str or pathlib.Path or dict): path or url to coco file. dict if multitask
            container_sas_or_root_dir (str): container sas if resources are store in blob container, or a local dir
        """

        if not coco_file_path_or_url:
            return None

        assert isinstance(coco_file_path_or_url, dict)
        assert isinstance(self.data_type, dict)
        dataset_manifest_by_task = {k: CocoManifestAdaptorFactory.create(self.data_type[k]).create_dataset_manifest(coco_file_path_or_url[k], container_sas_or_root_dir)
                                    for k in coco_file_path_or_url}

        return generate_multitask_dataset_manifest(dataset_manifest_by_task)

    def get_images_and_categories(self, images_by_id, coco_manifest):
        pass
