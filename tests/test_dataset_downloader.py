import io
import json
import os
import pathlib
import unittest
from unittest.mock import ANY, MagicMock

from vision_datasets.common import DatasetDownloader, DatasetRegistry, DatasetTypes


class TestDatasetDownloader(unittest.TestCase):
    def test_use_latest_version(self):
        datasets = [
            {
                'name': 'dataset_name',
                'version': 42,
                'type': DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS.name,
                'root_folder': '',
                'train': {'index_path': '42.txt', 'files_for_local_usage': []},
                'test': {'index_path': 'val42.txt', 'files_for_local_usage': []}
            },
            {
                'name': 'dataset_name',
                'version': 2,
                'type': DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS.name,
                'root_folder': './',
                'train': {'index_path': '2.txt', 'files_for_local_usage': []},
                'test': {'index_path': 'val2.txt', 'files_for_local_usage': []}
            },
            {
                'name': 'dataset_name',
                'version': 4,
                'type': DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS.name,
                'root_folder': './',
                'train': {'index_path': '4.txt', 'files_for_local_usage': []},
                'test': {'index_path': 'val4.txt', 'files_for_local_usage': []}
            },
            {
                'name': 'dataset_name2',
                'version': 43,
                'type': DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS.name,
                'root_folder': './',
                'train': {'index_path': '43.txt', 'files_for_local_usage': []},
                'test': {'index_path': 'val43.txt', 'files_for_local_usage': []}
            }
        ]
        dataset_info = self._make_reg(datasets).get_dataset_info('dataset_name')
        downloader = self._make_downloader(dataset_info)
        downloader._download_files = MagicMock()
        downloader.download('dataset_name')
        downloader._download_files.assert_called_once_with({pathlib.Path('42.txt'), pathlib.Path('val42.txt')}, unittest.mock.ANY)

    def test_delete_temp_dir(self):
        datasets = [{'name': 'dataset_name', 'type': DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS.name, 'root_folder': './', 'version': 42,
                     'train': {'index_path': '42.txt', 'files_for_local_usage': []},
                     'test': {'index_path': '42.txt', 'files_for_local_usage': []}}]

        dataset_info = self._make_reg(datasets).get_dataset_info('dataset_name')
        downloader = self._make_downloader(dataset_info)
        downloader._download_files = MagicMock()
        with downloader.download() as downloaded:
            for x in downloaded.base_dirs:
                self.assertTrue(os.path.isdir(x))

        for x in downloaded.base_dirs:
            self.assertFalse(os.path.isdir(x))

    def test_concatenate_path(self):
        datasets = [{
            'name': 'dataset_name',
            'type': DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS.name,
            'root_folder': 'somewhere',
            'version': 1,
            'train': {'index_path': 'dir/42.txt', 'files_for_local_usage': []}}]
        dataset_info = self._make_reg(datasets).get_dataset_info('dataset_name')
        downloader = self._make_downloader(dataset_info)
        with unittest.mock.patch('requests.get') as mock_get:
            mock_get.return_value.__enter__.return_value.raw = io.BytesIO(b'42')
            mock_get.return_value.__enter__.return_value.status_code = 200
            downloader.download()
            mock_get.assert_called_once_with('http://example.com/somewhere/dir/42.txt?sastoken=something', allow_redirects=True, stream=True, timeout=ANY)

    @staticmethod
    def _make_downloader(dataset_info, base_path='http://example.com/?sastoken=something'):
        return DatasetDownloader(base_path, dataset_info)

    @staticmethod
    def _make_reg(datasets) -> DatasetRegistry:
        return DatasetRegistry(json.dumps(datasets))


if __name__ == '__main__':
    unittest.main()
