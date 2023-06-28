import pathlib
import typing

from ..common import DatasetTypes, CocoManifestAdaptorBase, CocoManifestAdaptorFactory
from ..common.data_manifest.utils import generate_multitask_dataset_manifest


@CocoManifestAdaptorFactory.register(DatasetTypes.MULTITASK)
class MultiTaskCocoManifestAdaptor(CocoManifestAdaptorBase):
    def create_dataset_manifest(self, coco_file_path_or_url: typing.Union[str, dict, pathlib.Path], container_sas_or_root_dir: str = None):
        """ construct a dataset manifest out of coco file
        Args:
            coco_file_path_or_url (str or pathlib.Path or dict): path or url to coco file. dict if multitask
            container_sas_or_root_dir (str): container sas if resources are store in blob container, or a local dir
        """

        if not coco_file_path_or_url:
            return None

        if not isinstance(coco_file_path_or_url, dict):
            raise ValueError
        if not isinstance(self.data_type, dict):
            raise ValueError
        dataset_manifest_by_task = {k: CocoManifestAdaptorFactory.create(self.data_type[k]).create_dataset_manifest(coco_file_path_or_url[k], container_sas_or_root_dir)
                                    for k in coco_file_path_or_url}

        return generate_multitask_dataset_manifest(dataset_manifest_by_task)

    def get_images_and_categories(self, images_by_id, coco_manifest):
        pass
