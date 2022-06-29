import copy
import pathlib
import tempfile
import unittest
import zipfile
import json

from PIL import Image
import numpy as np

from tests.test_fixtures import DetectionTestFixtures
from vision_datasets import IrisManifestAdaptor, DatasetInfo, ManifestDataset, CocoManifestAdaptor
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

        Image.new('RGB', (100, 100)).save(pathlib.Path(tempdir.name) / '0.jpg')
        Image.new('RGB', (100, 100)).save(pathlib.Path(tempdir.name) / '1.jpg')
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

    def test_works_with_empty_manifest(self):
        dataset_manifest = DetectionTestFixtures.create_an_od_manifest()
        dataset_manifest.images = []
        assert len(ManifestDataset(DatasetInfo(DetectionTestFixtures.DATASET_INFO_DICT), dataset_manifest)) == 0


class TestCocoManifestDataset(unittest.TestCase):
    DATASET_INFO_DICT = {
        "name": "dummy",
        "version": 1,
        "type": "image_matting",
        "description": "A dummy test dataset",
        "format": "coco",
        "root_folder": "dummy",
        "test": {
            "index_path": "test.json",
            "files_for_local_usage": [
                "test.zip",
                "mask.zip"
            ],
            "num_images": 2
        },
    }

    MATTING_JSON_COCO_FORMAT = {
        "images": [{"id": 1, "file_name": "0.jpg", "zip_file": "test.zip"},
                   {"id": 2, "file_name": "test.zip@1.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 1, "label": "0.png", "zip_file": "mask.zip"},
            {"id": 2, "image_id": 2, "label": "mask.zip@1.png"},
        ]
    }

    @staticmethod
    def _create_an_image_matting_dataset():
        dataset_dict = copy.deepcopy(TestCocoManifestDataset.DATASET_INFO_DICT)
        tempdir = tempfile.TemporaryDirectory()
        dataset_dict['root_folder'] = tempdir.name
        images = [Image.new('RGB', (100, 100)), Image.new('RGB', (100, 100)), Image.new('L', (100, 100)), Image.new('L', (100, 100))]
        images[0].save(pathlib.Path(tempdir.name) / '0.jpg')
        images[1].save(pathlib.Path(tempdir.name) / '1.jpg')
        images[2].save(pathlib.Path(tempdir.name) / '0.png')
        images[3].save(pathlib.Path(tempdir.name) / '1.png')

        with zipfile.ZipFile(pathlib.Path(tempdir.name) / 'test.zip', 'w') as zf:
            zf.write(pathlib.Path(tempdir.name) / '0.jpg', '0.jpg')
            zf.write(pathlib.Path(tempdir.name) / '1.jpg', '1.jpg')

        with zipfile.ZipFile(pathlib.Path(tempdir.name) / 'mask.zip', 'w') as zf:
            zf.write(pathlib.Path(tempdir.name) / '0.png', '0.png')
            zf.write(pathlib.Path(tempdir.name) / '1.png', '1.png')

        with open(pathlib.Path(tempdir.name) / 'test.json', 'w') as f:
            json.dump(TestCocoManifestDataset.MATTING_JSON_COCO_FORMAT, f)

        dataset_info = DatasetInfo(dataset_dict)
        dataset_manifest = CocoManifestAdaptor.create_dataset_manifest(dataset_info.test_path, dataset_info.type, dataset_info.root_folder)
        dataset = ManifestDataset(dataset_info, dataset_manifest)
        return dataset, tempdir, images

    def test_image_matting_manifest(self):
        dataset, tempdir, images = self._create_an_image_matting_dataset()
        with tempdir:
            self.assertEqual(len(dataset), 2)
            image0, target0, _ = dataset[0]
            image1, target1, _ = dataset[1]
            self.assertEqual(list(image0.getdata()), list(images[0].getdata()))
            self.assertEqual(list(image1.getdata()), list(images[1].getdata()))
            self.assertTrue(np.array_equal(target0[0], np.asarray(images[2]), equal_nan=True))
            self.assertTrue(np.array_equal(target1[0], np.asarray(images[3]), equal_nan=True))
