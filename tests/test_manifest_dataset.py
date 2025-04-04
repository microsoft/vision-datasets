import copy
import json
import pathlib
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from tests.test_fixtures import DetectionTestFixtures
from vision_datasets.common import CocoManifestAdaptorFactory, DatasetInfo, DatasetInfoFactory, DatasetTypes, Usages, VisionDataset
from vision_datasets.common.data_manifest.iris_data_manifest_adaptor import IrisManifestAdaptor

from .resources.util import coco_database, schema_database


class TestVisionDataset(unittest.TestCase):
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
        dataset_dict = copy.deepcopy(TestVisionDataset.DATASET_INFO_DICT)
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
        dataset_manifest = IrisManifestAdaptor.create_dataset_manifest(dataset_info, Usages.TEST)

        dataset = VisionDataset(dataset_info, dataset_manifest, 'relative')
        return dataset, tempdir

    def test_od_manifest_with_different_coordinate_formats(self):
        dataset, tempdir = self._create_an_od_dataset()
        with tempdir:
            self.assertEqual(len(dataset), 2)
            self.assertEqual(len(dataset.categories), 4)
            image0, target0, _ = dataset[0]
            image1, target1, _ = dataset[1]
            self.assertEqual([label.label_data for label in target0], [[0, 0.0, 0.0, 1.0, 1.0], [1, 0.1, 0.1, 0.5, 1.0]])
            self.assertEqual([label.label_data for label in target1], [[1, 0.5, 0.5, 0.8, 0.8], [3, 0.0, 0.5, 1.0, 1.0]])
            dataset = VisionDataset(dataset.dataset_info, dataset.dataset_manifest, 'absolute')
            image0, target0, _ = dataset[0]
            image1, target1, _ = dataset[1]
            self.assertEqual([label.label_data for label in target0], [[0, 0.0, 0.0, 100.0, 100.0], [1, 10.0, 10.0, 50.0, 100.0]])
            self.assertEqual([label.label_data for label in target1], [[1, 50.0, 50.0, 80.0, 80.0], [3, 0.0, 50.0, 100.0, 100.0]])

    def test_works_with_empty_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_manifest = DetectionTestFixtures.create_an_od_manifest(temp_dir)
            dataset_manifest.images = []
            self.assertEqual(len(VisionDataset(DatasetInfo(DetectionTestFixtures.DATASET_INFO_DICT), dataset_manifest)), 0)


class TestCocoVisionDataset(unittest.TestCase):
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
        dataset_dict = copy.deepcopy(TestCocoVisionDataset.DATASET_INFO_DICT)
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
            json.dump(TestCocoVisionDataset.MATTING_JSON_COCO_FORMAT, f)

        dataset_info = DatasetInfo(dataset_dict)
        dataset_manifest = CocoManifestAdaptorFactory.create(dataset_info.type).create_dataset_manifest(dataset_info.index_files[Usages.TEST], dataset_info.root_folder)
        dataset = VisionDataset(dataset_info, dataset_manifest)
        return dataset, tempdir, images

    def test_image_matting_manifest(self):
        dataset, tempdir, images = self._create_an_image_matting_dataset()
        with tempdir:
            self.assertEqual(len(dataset), 2)
            image0, target0, _ = dataset[0]
            image1, target1, _ = dataset[1]
            self.assertEqual(list(image0.getdata()), list(images[0].getdata()))
            self.assertEqual(list(image1.getdata()), list(images[1].getdata()))
            self.assertTrue(np.array_equal(target0[0].label_data, np.asarray(images[2]), equal_nan=True))
            self.assertTrue(np.array_equal(target1[0].label_data, np.asarray(images[3]), equal_nan=True))


class TestCocoKeyValuePairDataset(unittest.TestCase):
    key_value_pair_coco = coco_database[DatasetTypes.KEY_VALUE_PAIR][1]
    schema = schema_database[1]

    DATASET_INFO_DICT = {
        "name": "dummy",
        "version": 1,
        "type": "key_value_pair",
        "description": "A dummy test dataset",
        "format": "coco",
        "root_folder": "dummy",
        "test": {
            "index_path": "test.json",
            "files_for_local_usage": [
                "test.zip",
            ],
            "num_images": 2
        },
        "schema": schema
    }

    @staticmethod
    def _create_key_value_pair_dataset(tempdir):
        dataset_dict = copy.deepcopy(TestCocoKeyValuePairDataset.DATASET_INFO_DICT)
        dataset_dict['root_folder'] = tempdir
        images = [Image.new('RGB', (100, 100)), Image.new('RGB', (50, 50))]
        images[0].save(pathlib.Path(tempdir) / '0.jpg')
        images[1].save(pathlib.Path(tempdir) / '1.jpg')

        with zipfile.ZipFile(pathlib.Path(tempdir) / 'test.zip', 'w') as zf:
            zf.write(pathlib.Path(tempdir) / '0.jpg', '0.jpg')
            zf.write(pathlib.Path(tempdir) / '1.jpg', '1.jpg')

        with open(pathlib.Path(tempdir) / 'test.json', 'w') as f:
            json.dump(TestCocoKeyValuePairDataset.key_value_pair_coco, f)

        dataset_info = DatasetInfoFactory.create(dataset_dict)
        schema = dataset_info.schema
        # Alternatively, provide schema dictionary directly:
        # schema = TestCocoKeyValuePairDataset.DATASET_INFO_DICT['schema']

        dataset_manifest = CocoManifestAdaptorFactory.create(dataset_info.type, schema=schema).create_dataset_manifest(dataset_info.index_files[Usages.TEST], dataset_info.root_folder)
        dataset = VisionDataset(dataset_info, dataset_manifest)
        return dataset, images

    def test_key_value_pair_dataset(self):
        with tempfile.TemporaryDirectory() as tempdir:
            dataset, images = self._create_key_value_pair_dataset(tempdir)
            self.assertEqual(len(dataset), 2)
            images_0, target0, _ = dataset[0]
            self.assertEqual(list(images_0[0].getdata()), list(images[0].getdata()))
            self.assertEqual(list(images_0[1].getdata()), list(images[1].getdata()))
            self.assertEqual(target0.fields, self.key_value_pair_coco['annotations'][0]['fields'])
            self.assertEqual(target0.text, None)

            images_1, target1, _ = dataset[1]
            self.assertEqual(list(images_1[0].getdata()), list(images[1].getdata()))
            self.assertEqual(list(images_1[1].getdata()), list(images[0].getdata()))
            self.assertEqual(target1.fields, self.key_value_pair_coco['annotations'][1]['fields'])
            self.assertEqual(target1.text, {"note": "reversed image order."})

            # test get targets
            for i, coco_anno in enumerate(self.key_value_pair_coco['annotations']):
                t = dataset.get_targets(i)
                self.assertEqual(t.fields, coco_anno['fields'])
                self.assertEqual(t.text, coco_anno.get('text', None))
