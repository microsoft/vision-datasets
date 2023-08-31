import logging
from typing import List, Union, Tuple

from ..constants import Usages
from ..data_manifest import ManifestMerger, DatasetManifest
from ..dataset_info import MultiTaskDatasetInfo, BaseDatasetInfo
from ..factory import DataManifestFactory, ManifestMergeStrategyFactory
from ..dataset import VisionDataset
from ..data_reader import DatasetDownloader, DownloadedDatasetsResources
from .dataset_registry import DatasetRegistry

logger = logging.getLogger(__name__)


class DatasetHub(object):
    """
    Hub class for managing vision dataset resources, with a few common utilities for creating a dataset.
    This hub class works with both resources on local disk or on azure blob.
    """

    def __init__(self, dataset_json_str: Union[str, list], container_url: str, local_dir: str):
        """
            If local_dir is provided, manifest_dataset consumes data from local disk. If data not present on local disk, it will be automatically downloaded.
            if container_url is provided but local_dir not provided, manifest_dataset consumes data directly from container_url.
            Note that for data stored in zipped files, they can be consumed locally without unzip. However, in blob they must be stored in unzipped folders. In this case image/label file paths can
            stay with paths to data in zipped files, as dataset class will automatically look in the folder names same with the zip file names.
        Args:
            dataset_json_str (str, list): dataset registry json, containing multiple dataset_info for different datasets, or a list of dataset reg json
                retrievable by their names, versions and usages.
            container_url (str): sas url to the container where datasets can be found/downloaded from
            local_dir (str): local directory where datasets can be found/downloaded to
        """
        if not dataset_json_str:
            raise ValueError
        if not container_url and not local_dir:
            raise ValueError('either container_url or local_dir should be provided.')
        self.dataset_registry = DatasetRegistry(dataset_json_str)
        self.container_url = container_url
        self.local_dir = local_dir

    def create_vision_dataset(self, name: str, version: int = None, usage: Union[str, List] = Usages.TRAIN, coordinates: str = 'relative') -> VisionDataset:
        """Create manifest dataset.

            Note that for data stored in zipped files, they can be consumed locally without unzip. However, in blob they must be stored in unzipped folders. In this case image/label file paths can
            stay with paths to data in zipped files, as dataset class will automatically look in the folder names same with the zip file names.

        Args:
            name: dataset name
            version: dataset version, if not specified, latest version will be returned
            usage: usage(s) of the dataset, 'train', 'val' or 'test' or a list of usages
            coordinates: format of the bounding boxes, can be 'relative' or 'absolute'

        Returns:
            an instance of dataset for local usage
        """
        manifest, dataset_info, downloader_resources = self.create_dataset_manifest(name, version, usage)
        if manifest is None:
            return None

        return VisionDataset(dataset_info, manifest, coordinates, downloader_resources)

    def create_dataset_manifest(self, name: str, version: int = None, usage: Union[str, List] = Usages.TRAIN) -> Tuple[DatasetManifest, BaseDatasetInfo, DownloadedDatasetsResources]:
        """Create dataset manifest.

        Args:
            name: dataset name
            version: dataset version, if not specified, latest version will be returned
            usage: usage(s) of the dataset, 'train', 'val' or 'test' or a list of usages

        Returns:
            dataset manifest, dataset_info, downloaded_resources, if dataset exists, else None
        """
        if not name:
            raise ValueError
        if not usage:
            raise ValueError

        dataset_info = self.dataset_registry.get_dataset_info(name, version)
        if dataset_info is None:
            logger.warning(f'Dataset with {name} and version {version} not found.')
            return None, None, None

        usages = usage if isinstance(usage, list) else [usage]
        if isinstance(dataset_info, MultiTaskDatasetInfo):
            for task_info in dataset_info.sub_task_infos.values():
                task_info.index_files = {usage: task_info.index_files[usage] for usage in usages if usage in task_info.index_files}
        else:
            dataset_info.index_files = {usage: dataset_info.index_files[usage] for usage in usages if usage in dataset_info.index_files}

        downloader_resources = None
        if self.container_url and self.local_dir:
            downloader = DatasetDownloader(self.container_url, self.dataset_registry.get_dataset_info(name, version))
            downloader_resources_usage = downloader.download(self.local_dir, usages)
        else:
            downloader_resources_usage = None

        manifest = None
        for usage in usages:
            manifest_usage = DataManifestFactory.create(dataset_info, usage, self.local_dir or self.container_url)
            if manifest_usage is not None:
                merger = ManifestMerger(ManifestMergeStrategyFactory.create(dataset_info.type))
                manifest = merger.run(manifest, manifest_usage) if manifest else manifest_usage

            if downloader_resources_usage:
                downloader_resources = DownloadedDatasetsResources.merge(downloader_resources, downloader_resources_usage) if downloader_resources else downloader_resources_usage
        if manifest is None:
            logger.warning(f'Dataset with {name}, version {version}, and usage(s) {usages} not found.')
            return None, None, None

        return manifest, dataset_info, downloader_resources

    def list_data_version_and_types(self):
        """List all dataset names, versions and types
        """

        return self.dataset_registry.list_data_version_and_types()
