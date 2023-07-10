import logging
import os
import pathlib
import re
import shutil
import tempfile
from typing import List
from urllib import parse as urlparse

import azure.storage.blob
import requests
import tenacity
from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential

from ..constants import DatasetTypes, Usages
from ..dataset_info import BaseDatasetInfo, DatasetInfo
from ..utils import can_be_url

logger = logging.getLogger(__name__)


class AzureDownloader:
    def __init__(self, container_url: str) -> None:
        self._container_url = container_url
        has_sas = 'sig=' in container_url
        credential = None if has_sas else DefaultAzureCredential()
        self._container_client = azure.storage.blob.ContainerClient.from_container_url(container_url, credential=credential)

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_if_exception_type(AzureError), reraise=True)
    def download(self, file_path, target_dir):
        target_dir = pathlib.Path(target_dir)
        (target_dir / file_path).parent.mkdir(parents=True, exist_ok=True)
        stream = self._container_client.download_blob(file_path, max_concurrency=8, read_timeout=1800)
        with open((target_dir / file_path), 'wb') as f:
            stream.readinto(f)

    @staticmethod
    def is_azure_blob_url(url):
        return 'blob.core.windows.net' in url


class DownloadedDatasetsResources:
    """Wrapper class to make sure the temporary directory is removed."""

    def __init__(self, base_dirs: List[pathlib.Path]):
        self.base_dirs = base_dirs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for base_dir in self.base_dirs:
            if os.path.isdir(base_dir):
                logger.info(f'Removing folder: {base_dir}.')
                shutil.rmtree(base_dir)

    @staticmethod
    def merge(r1, r2):
        if r1 is None or r2 is None:
            raise ValueError

        return DownloadedDatasetsResources(r1.base_dirs + r2.base_dirs)


class DatasetDownloader:
    def __init__(self, dataset_sas_url: str, dataset_info: BaseDatasetInfo):
        if not dataset_info:
            raise ValueError

        if not can_be_url(dataset_sas_url):
            raise ValueError('An url to the dataset should be provided.')

        self._base_url = dataset_sas_url
        self._dataset_info = dataset_info

    def download(self, target_dir: str = None, purposes=[Usages.TRAIN, Usages.VAL, Usages.TEST]):
        if not purposes:
            raise ValueError

        target_dir = pathlib.Path(tempfile.mkdtemp()) if target_dir is None else pathlib.Path(target_dir)
        (target_dir / pathlib.Path(self._dataset_info.root_folder)).mkdir(parents=True, exist_ok=True)

        if self._dataset_info.type == DatasetTypes.MULTITASK:
            files_to_download = set.union(*[self._find_files_to_download(subtask_info, purposes) for subtask_info in self._dataset_info.sub_task_infos.values()])
        else:
            files_to_download = self._find_files_to_download(self._dataset_info, purposes)

        self._download_files(files_to_download, target_dir)

        return DownloadedDatasetsResources([target_dir])

    @staticmethod
    def _keep_until_including_pattern(s, pattern):
        match = re.search(pattern, s)
        if match:
            end = match.end()
            return s[:end]
        else:
            return s

    def _find_files_to_download(self, dataset_info: DatasetInfo, purposes: List[str]) -> set:
        files_to_download = set()
        rt_dir = pathlib.Path(dataset_info.root_folder)
        for usage in purposes:
            if usage in dataset_info.index_files:
                # index file can be included in a zip file as well, e.g., "index_files.zip@ann.json"
                file = self._keep_until_including_pattern(dataset_info.index_files[usage], pattern=r'@*\.zip')
                files_to_download.add(rt_dir / file)
            if usage in dataset_info.files_for_local_usage:
                files_to_download.update([rt_dir / x for x in dataset_info.files_for_local_usage[usage]])

        if dataset_info.labelmap:
            files_to_download.add(rt_dir / dataset_info.labelmap)

        if dataset_info.image_metadata_path:
            files_to_download.add(rt_dir / dataset_info.image_metadata_path)

        return files_to_download

    def _download_files(self, file_paths: List, target_dir: pathlib.Path):
        parts = urlparse.urlparse(self._base_url)

        azure_downloader = AzureDownloader(self._base_url) if AzureDownloader.is_azure_blob_url(self._base_url) else None
        for file_path in file_paths:
            path = os.path.join(parts[2], file_path).replace('\\', '/')
            url = urlparse.urlunparse((parts[0], parts[1], path, parts[3], parts[4], parts[5]))
            target_file_path = target_dir / file_path
            target_file_path.parent.mkdir(parents=True, exist_ok=True)

            if target_file_path.exists():
                logger.info(f'{target_file_path} exists. Skip downloading.')
                continue

            if AzureDownloader.is_azure_blob_url(url):
                try:
                    logger.info('Detected the URL is from Azure blob.')
                    azure_downloader.download(file_path.as_posix(), target_dir)
                except Exception as e:
                    logger.warn(f'Azure downloading fails {e}. Fallback to regular download.')
                    self._download_file(url, target_file_path)
            else:
                self._download_file(url, target_file_path)

    @tenacity.retry(stop=tenacity.stop_after_attempt(3))
    def _download_file(self, url: str, filepath: pathlib.Path):
        logger.info(f'Downloading from {url} to {filepath.absolute()}.')
        with requests.get(url, stream=True, allow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                shutil.copyfileobj(r.raw, f, length=4194304)
