import copy
import os
import tempfile
import unittest

from PIL import Image
from vision_datasets import IrisManifestAdaptor, DatasetInfo, ManifestDataset
from vision_datasets.common.constants import Usages


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

    def test_od_manifest_with_different_coordinate_formats(self):
        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            dataset_dict['root_folder'] = str(tempdir)
            dataset_dict['type'] = 'object_detection'
            with open(os.path.join(tempdir, 'test.txt'), 'w') as f:
                f.write('0.jpg 0.txt\n1.jpg 1.txt')

            with open(os.path.join(tempdir, '0.txt'), 'w') as f:
                f.write('0 0 100 0 100\n1 0 100 0 100')

            with open(os.path.join(tempdir, '1.txt'), 'w') as f:
                f.write('1 50 50 100 100\n3 0 50 100 100')

            Image.new('RGB', (100, 100)).save(os.path.join(tempdir, '0.jpg'))
            Image.new('RGB', (100, 100)).save(os.path.join(tempdir, '1.jpg'))
            dataset_info = DatasetInfo(dataset_dict)
            dataset_manifest = IrisManifestAdaptor.create_dataset_manifest(dataset_info, Usages.TEST_PURPOSE)

            dataset = ManifestDataset(dataset_info, dataset_manifest, 'relative')
            self.assertEqual(len(dataset), 2)
            self.assertEqual(len(dataset.labels), 4)
            image0, target0, _ = dataset[0]
            image1, target1, _ = dataset[1]
            self.assertEqual(target0, [[0, 0.0, 1.0, 0.0, 1.0], [1, 0.0, 1.0, 0.0, 1.0]])
            self.assertEqual(target1, [[1, 0.5, 0.5, 1.0, 1.0], [3, 0.0, 0.5, 1.0, 1.0]])
            dataset = ManifestDataset(dataset_info, dataset_manifest, 'absolute')
            image0, target0, _ = dataset[0]
            image1, target1, _ = dataset[1]
            self.assertEqual(target0, [[0, 0.0, 100.0, 0.0, 100.0], [1, 0.0, 100.0, 0.0, 100.0]])
            self.assertEqual(target1, [[1, 50.0, 50.0, 100.0, 100.0], [3, 0.0, 50.0, 100.0, 100.0]])
