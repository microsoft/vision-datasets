import io
import json
import os
import unittest
from unittest.mock import MagicMock, ANY

from vision_datasets import DatasetTypes
from vision_datasets.common.dataset_downloader import DatasetDownloader


class TestDatasetDownloader(unittest.TestCase):
    def test_no_entry(self):
        datasets = []
        downloader = self._make_downloader(datasets)
        with self.assertRaises(RuntimeError):
            downloader.download('dataset_name')

    def test_no_version(self):
        datasets = [{
            'name': 'dataset_name',
            'version': 42,
            'type': DatasetTypes.IC_MULTICLASS,
            'root_folder': '',
            'train': {'index_path': '42.txt', 'files_for_local_usage': []},
            'test': {'index_path': 'val42.txt', 'files_for_local_usage': []}
        }]
        downloader = self._make_downloader(datasets)
        with self.assertRaises(RuntimeError):
            downloader.download('dataset_name', 3)

    def test_use_latest_version(self):
        datasets = [
            {
                'name': 'dataset_name',
                'version': 42,
                'type': DatasetTypes.IC_MULTICLASS,
                'root_folder': '',
                'train': {'index_path': '42.txt', 'files_for_local_usage': []},
                'test': {'index_path': 'val42.txt', 'files_for_local_usage': []}
            },
            {
                'name': 'dataset_name',
                'version': 2,
                'type': DatasetTypes.IC_MULTICLASS,
                'root_folder': './',
                'train': {'index_path': '2.txt', 'files_for_local_usage': []},
                'test': {'index_path': 'val2.txt', 'files_for_local_usage': []}
            },
            {
                'name': 'dataset_name',
                'version': 4,
                'type': DatasetTypes.IC_MULTICLASS,
                'root_folder': './',
                'train': {'index_path': '4.txt', 'files_for_local_usage': []},
                'test': {'index_path': 'val4.txt', 'files_for_local_usage': []}
            },
            {
                'name': 'dataset_name2',
                'version': 43,
                'type': DatasetTypes.IC_MULTICLASS,
                'root_folder': './',
                'train': {'index_path': '43.txt', 'files_for_local_usage': []},
                'test': {'index_path': 'val43.txt', 'files_for_local_usage': []}
            }
        ]
        downloader = self._make_downloader(datasets)
        downloader._download_files = MagicMock()
        downloader.download('dataset_name')
        downloader._download_files.assert_called_once_with({'42.txt', 'val42.txt'}, unittest.mock.ANY)

    def test_delete_temp_dir(self):
        datasets = [{'name': 'dataset_name', 'type': DatasetTypes.IC_MULTICLASS, 'root_folder': './', 'version': 42,
                     'train': {'index_path': '42.txt', 'files_for_local_usage': []},
                     'test': {'index_path': '42.txt', 'files_for_local_usage': []}}]

        downloader = self._make_downloader(datasets)
        downloader._download_files = MagicMock()
        with downloader.download('dataset_name') as downloaded:
            for x in downloaded.base_dirs:
                self.assertTrue(os.path.isdir(x))

        for x in downloaded.base_dirs:
            self.assertFalse(os.path.isdir(x))

    def test_concatenate_path(self):
        datasets = [{'name': 'dataset_name', 'type': DatasetTypes.IC_MULTICLASS, 'root_folder': 'somewhere', 'version': 1, 'train': {'index_path': 'dir/42.txt', 'files_for_local_usage': []}}]

        downloader = self._make_downloader(datasets)
        with unittest.mock.patch('requests.get') as mock_get:
            mock_get.return_value.__enter__.return_value.raw = io.BytesIO(b'42')
            mock_get.return_value.__enter__.return_value.status_code = 200
            downloader.download('dataset_name')
            mock_get.assert_called_once_with('http://example.com/somewhere/dir/42.txt?sastoken=something', allow_redirects=True, stream=True, timeout=ANY)

    @staticmethod
    def _make_downloader(datasets, base_path='http://example.com/?sastoken=something'):
        from vision_datasets.common.dataset_registry import DatasetRegistry

        return DatasetDownloader(base_path, DatasetRegistry(json.dumps(datasets)))


if __name__ == '__main__':
    unittest.main()
