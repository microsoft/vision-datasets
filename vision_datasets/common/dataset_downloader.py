import logging
import os
import pathlib
from typing import List

import requests
import shutil
import tempfile
import tenacity
from urllib import parse as urlparse

from .dataset_registry import DatasetRegistry
from .dataset_info import DatasetInfo, DatasetInfoFactory
from .constants import Usages
from .util import is_url

logger = logging.getLogger(__name__)


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


class DatasetDownloader:
    def __init__(self, dataset_sas_url: str, dataset_registry: DatasetRegistry):
        assert dataset_sas_url
        assert dataset_registry

        if not is_url(dataset_sas_url):
            raise RuntimeError('An url to the dataset should be provided.')

        self._dataset_sas_url = dataset_sas_url
        self._registry = dataset_registry

    def download(self, name: str, version: int = None, target_dir: str = None, purposes=[Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE]):
        assert purposes

        dataset_info = self._registry.get_dataset_info(name, version)
        if not dataset_info:
            raise RuntimeError(f'No dataset matched for the specified condition: {name} ({version})')

        target_dir = pathlib.Path(tempfile.mkdtemp()) if target_dir is None else pathlib.Path(target_dir)
        target_dir = target_dir / pathlib.Path(dataset_info.root_folder)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if DatasetInfoFactory.is_multitask(dataset_info.type):
            for subtask_info in dataset_info.sub_task_infos.values():
                self._download(subtask_info, target_dir, purposes)
        else:
            self._download(dataset_info, target_dir, purposes)

        return DownloadedDatasetsResources([target_dir])

    def _download(self, dataset_info: DatasetInfo, target_dir, purposes):
        files_to_download = set()

        for usage in purposes:
            if usage in dataset_info.index_files:
                files_to_download.add(os.path.join(dataset_info.root_folder, dataset_info.index_files[usage]))
            if usage in dataset_info.files_for_local_usage:
                files_to_download.update([os.path.join(dataset_info.root_folder, x) for x in dataset_info.files_for_local_usage[usage]])

        if dataset_info.labelmap:
            files_to_download.add(os.path.join(dataset_info.root_folder, dataset_info.labelmap))

        if dataset_info.image_metadata_path:
            files_to_download.add(os.path.join(dataset_info.root_folder, dataset_info.image_metadata_path))

        self._download_files(files_to_download, target_dir)

    def _download_files(self, file_paths, target_dir: pathlib.Path):
        parts = urlparse.urlparse(self._dataset_sas_url)
        for file_path in file_paths:
            path = os.path.join(parts[2], file_path).replace('\\', '/')
            url = urlparse.urlunparse((parts[0], parts[1], path, parts[3], parts[4], parts[5]))
            target_file_path = target_dir / pathlib.Path(file_path).name
            if os.path.exists(target_file_path):
                logger.info(f'{target_file_path} exists. Skip downloading.')
                continue

            self._download_file(url, target_file_path)

    @tenacity.retry(stop=tenacity.stop_after_attempt(3))
    def _download_file(self, url, filepath):
        logger.info(f'Downloading from {url} to {filepath.absolute()}.')
        with requests.get(url, stream=True, allow_redirects=True, timeout=60) as r:
            if r.status_code > 200:
                raise RuntimeError(f'Failed in downloading from {url}, status code {r.status_code}.')

            with open(filepath, 'wb') as f:
                shutil.copyfileobj(r.raw, f, length=4194304)
