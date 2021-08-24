import logging

from ..common import DatasetRegistry, Usages
from ..common.data_manifest import DatasetManifest
from ..common.dataset_info import MultiTaskDatasetInfo

logger = logging.getLogger(__name__)


class DatasetHub(object):
    """
    Hub class for managing vision dataset resources, with a few common utilities for creating a dataset.
    This hub class works with both resources on local disk or on azure blob.
    """

    def __init__(self, dataset_json_str):
        """

        Args:
            dataset_json_str (str): dataset registry json, containing multiple dataset_info for different datasets,
            retrievable by their names, versions and usages.
        """
        assert dataset_json_str

        self.dataset_registry = DatasetRegistry(dataset_json_str)

    def create_manifest_dataset(self, container_sas: str, local_dir: str, name: str, version: int = None, usage: str = Usages.TRAIN_PURPOSE, coordinates: str = 'relative',
                                few_shot_samples_per_class=None, rnd_seed=0):
        """Create manifest dataset.
            If local_dir is provided, manifest_dataset consumes data from local disk. If data not present on local disk, it will be automatically downloaded.
            if container_sas is provided but local_dir not provided, manifest_dataset consumes data directly from container_sas.

            Note that for data stored in zipped files, they can be consumed locally without unzip. However, in blob they must be stored in unzipped folders. In this case image/label file paths can
            stay with paths to data in zipped files, as dataset class will automatically look in the folder names same with the zip file names.

        Args:
            container_sas: sas url to the container where datasets can be found/downloaded from
            local_dir: local directory where datasets can be found/downloaded to
            name: dataset name
            version: dataset version, if not specified, latest version will be returned
            usage: usage of the dataset, 'train', 'val' or 'test'
            coordinates: format of the bounding boxes, can be 'relative' or 'absolute'
            few_shot_samples_per_class (int): get a sampled dataset with N images at most for each class (for detection and multilabel case, not guaranteed.)
            rnd_seed (int): random seed for few shot sampling

        Returns:
            an instance of dataset for local usage
        """
        result = self.create_dataset_manifest(container_sas, local_dir, name, version, usage, few_shot_samples_per_class, rnd_seed)
        if result:
            manifest, dataset_info, downloader_resources = result
        else:
            return None

        from vision_datasets import ManifestDataset
        return ManifestDataset(dataset_info, manifest, coordinates, downloader_resources)

    def create_dataset_manifest(self, container_sas: str, local_dir: str, name: str, version: int = None, usage: str = Usages.TRAIN_PURPOSE, few_shot_samples_per_class=None, rnd_seed=0):
        """Create dataset manifest.
            If local_dir is provided, manifest_dataset consumes data from local disk. If data not present on local disk, it will be automatically downloaded.
            if container_sas is provided but local_dir not provided, manifest_dataset consumes data directly from container_sas.

            Note that for data stored in zipped files, they can be consumed locally without unzip. However, in blob they must be stored in unzipped folders. In this case image/label file paths can
            stay with paths to data in zipped files, as dataset class will automatically look in the folder names same with the zip file names.

        Args:
            container_sas: sas url to the container where datasets can be found/downloaded from
            local_dir: local directory where datasets can be found/downloaded to
            name: dataset name
            version: dataset version, if not specified, latest version will be returned
            usage: usage of the dataset, 'train', 'val' or 'test'
            few_shot_samples_per_class (int): get a sampled dataset with N images at most for each class (for detection and multilabel case, not guaranteed.)
            rnd_seed (int): random seed for few shot sampling

        Returns:
            dataset manifest dataset, dataset_info, downloaded_resources, if dataset exists, else None
        """
        assert container_sas or local_dir
        assert name

        from ..common.dataset_downloader import DatasetDownloader

        dataset_info = self.dataset_registry.get_dataset_info(name, version)
        if dataset_info is None:
            logger.warning(f'Dataset with {name} and version {version} not found.')
            return None

        if isinstance(dataset_info, MultiTaskDatasetInfo):
            for task_info in dataset_info.sub_task_infos.values():
                task_info.index_files = {usage: task_info.index_files[usage]} if usage in task_info.index_files else {}
        else:
            dataset_info.index_files = {usage: dataset_info.index_files[usage]} if usage in dataset_info.index_files else {}

        if container_sas and local_dir:
            downloader = DatasetDownloader(container_sas, self.dataset_registry)
            downloader_resources = downloader.download(name, version, local_dir, [usage])
        else:
            downloader_resources = None

        manifest = DatasetManifest.create_dataset_manifest(dataset_info, usage, local_dir or container_sas)

        if not manifest:
            return None

        if few_shot_samples_per_class:
            original_img_cnt = len(manifest.images)
            manifest = manifest.sample_few_shot_subset(few_shot_samples_per_class, rnd_seed)
            logger.info(f'Create a few-shot dataset with n samples per class = {few_shot_samples_per_class}. # images: {original_img_cnt} => {len(manifest.images)}')

        return manifest, dataset_info, downloader_resources

    def list_data_version_and_types(self):
        """List all dataset names, versions and types
        """

        return self.dataset_registry.list_data_version_and_types()
