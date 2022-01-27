import copy
import os
import pathlib
import tempfile
import unittest

from PIL import Image
from vision_datasets import IrisManifestAdaptor, DatasetInfo, ManifestDataset
from vision_datasets.common.constants import Usages, DatasetTypes
from vision_datasets.common.manifest_dataset import DetectionAsClassificationDataset


class TestManifestDataset(unittest.TestCase):
    DATASET_INFO_DICT = {
        "name": "dummy",
        "version": 1,
        "type": "classification_multiclass",
        "root_folder": "dummy",
        "test": {
            "index_path": "test.txt",
            "files_for_local_usage": [
                "Train.zip"
            ]
        },
    }

    @staticmethod
    def _create_an_od_dataset():
        dataset_dict = copy.deepcopy(TestManifestDataset.DATASET_INFO_DICT)
        tempdir = tempfile.TemporaryDirectory()
        dataset_dict['root_folder'] = tempdir.name
        dataset_dict['type'] = 'object_detection'
        with open(pathlib.Path(tempdir.name) / 'test.txt', 'w') as f:
            f.write('0.jpg 0.txt\n1.jpg 1.txt')

        with open(pathlib.Path(tempdir.name) / '0.txt', 'w') as f:
            f.write('0 0 0 100 100\n1 10 10 50 100')

        with open(pathlib.Path(tempdir.name) / '1.txt', 'w') as f:
            f.write('1 50 50 80 80\n3 0 50 100 100')

        Image.new('RGB', (100, 100)).save(os.path.join(tempdir.name, '0.jpg'))
        Image.new('RGB', (100, 100)).save(os.path.join(tempdir.name, '1.jpg'))
        dataset_info = DatasetInfo(dataset_dict)
        dataset_manifest = IrisManifestAdaptor.create_dataset_manifest(dataset_info, Usages.TEST_PURPOSE)

        dataset = ManifestDataset(dataset_info, dataset_manifest, 'relative')
        return dataset, tempdir

    def test_od_manifest_with_different_coordinate_formats(self):
        dataset, tempdir = self._create_an_od_dataset()
        with tempdir:
            self.assertEqual(len(dataset), 2)
            self.assertEqual(len(dataset.labels), 4)
            image0, target0, _ = dataset[0]
            image1, target1, _ = dataset[1]
            self.assertEqual(target0, [[0, 0.0, 0.0, 1.0, 1.0], [1, 0.1, 0.1, 0.5, 1.0]])
            self.assertEqual(target1, [[1, 0.5, 0.5, 0.8, 0.8], [3, 0.0, 0.5, 1.0, 1.0]])
            dataset = ManifestDataset(dataset.dataset_info, dataset.dataset_manifest, 'absolute')
            image0, target0, _ = dataset[0]
            image1, target1, _ = dataset[1]
            self.assertEqual(target0, [[0, 0.0, 0.0, 100.0, 100.0], [1, 10.0, 10.0, 50.0, 100.0]])
            self.assertEqual(target1, [[1, 50.0, 50.0, 80.0, 80.0], [3, 0.0, 50.0, 100.0, 100.0]])

    def test_od_as_ic_dataset(self):
        dataset, tempdir = self._create_an_od_dataset()
        with tempdir:
            ic_dataset = DetectionAsClassificationDataset(dataset)
            assert ic_dataset.dataset_info.type == DatasetTypes.IC_MULTICLASS
            assert len(ic_dataset) == 4
            assert ic_dataset[0][0].size == (100, 100)
            assert ic_dataset[1][0].size == (40, 90)
            assert ic_dataset[2][0].size == (30, 30)
            assert ic_dataset[3][0].size == (100, 50)
