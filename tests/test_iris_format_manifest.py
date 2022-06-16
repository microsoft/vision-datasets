import copy
import os
import tempfile
import unittest
from unittest.mock import patch

from vision_datasets import IrisManifestAdaptor, DatasetInfo, DatasetManifest
from vision_datasets.common.constants import Usages


class TestCreateIrisDatasetManifest(unittest.TestCase):
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
        }
    }

    def test_detect_object_detection(self):
        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'test.txt'), 'w') as f:
                f.write("test.jpg l.txt")
            with open(os.path.join(tempdir, 'l.txt'), 'w') as f:
                f.write("0 1 2 3 4")
            dataset_dict['root_folder'] = str(tempdir)
            dataset_dict['type'] = 'object_detection'
            with patch('vision_datasets.common.data_manifest.DatasetManifest') as m:
                IrisManifestAdaptor.create_dataset_manifest(DatasetInfo(dataset_dict), Usages.TEST_PURPOSE)
                m.assert_called_once()

    def test_detect_multilabel(self):
        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'test.txt'), 'w') as f:
                f.write("test.jpg 2,3")
            dataset_dict['root_folder'] = str(tempdir)
            with patch('vision_datasets.common.data_manifest.DatasetManifest') as m:
                IrisManifestAdaptor.create_dataset_manifest(DatasetInfo(dataset_dict), Usages.TEST_PURPOSE)
                m.assert_called_once()

    def test_detect_multiclass(self):
        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'test.txt'), 'w') as f:
                f.write("test.jpg 2")
            dataset_dict['root_folder'] = str(tempdir)
            with patch('vision_datasets.common.data_manifest.DatasetManifest') as m:
                IrisManifestAdaptor.create_dataset_manifest(DatasetInfo(dataset_dict), Usages.TEST_PURPOSE)
                m.assert_called_once()

    def test_space_in_image_path(self):
        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'test.txt'), 'w') as f:
                f.write("test folder/0.jpg 0\n")
            dataset_dict['root_folder'] = str(tempdir)
            dataset_manifest = IrisManifestAdaptor.create_dataset_manifest(DatasetInfo(dataset_dict), Usages.TEST_PURPOSE)
            self.assertIsInstance(dataset_manifest, DatasetManifest)
            self.assertEqual(len(dataset_manifest.images), 1)
            self.assertEqual(len(dataset_manifest.labelmap), 1)
            self.assertEqual(dataset_manifest.images[0].id, 'test folder/0.jpg')
            self.assertEqual(dataset_manifest.images[0].labels, [0])

    def test_multiclass(self):
        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'test.txt'), 'w') as f:
                f.write("0.jpg 0\n1.jpg 1\n2.jpg 2")
            dataset_dict['root_folder'] = str(tempdir)
            dataset_manifest = IrisManifestAdaptor.create_dataset_manifest(DatasetInfo(dataset_dict), Usages.TEST_PURPOSE)
            self.assertIsInstance(dataset_manifest, DatasetManifest)
            self.assertEqual(len(dataset_manifest.images), 3)
            self.assertEqual(len(dataset_manifest.labelmap), 3)
            self.assertEqual(dataset_manifest.images[0].id, '0.jpg')
            self.assertEqual(dataset_manifest.images[0].labels, [0])
            self.assertEqual(dataset_manifest.images[1].id, '1.jpg')
            self.assertEqual(dataset_manifest.images[1].labels, [1])
            self.assertEqual(dataset_manifest.images[2].id, '2.jpg')
            self.assertEqual(dataset_manifest.images[2].labels, [2])

    def test_multilabel(self):
        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'test.txt'), 'w') as f:
                f.write("0.jpg 0,1\n1.jpg 1,2\n2.jpg 2")
            dataset_dict['root_folder'] = str(tempdir)
            dataset_dict['type'] = 'classification_multilabel'
            dataset_manifest = IrisManifestAdaptor.create_dataset_manifest(DatasetInfo(dataset_dict), Usages.TEST_PURPOSE)

            self.assertIsInstance(dataset_manifest, DatasetManifest)
            self.assertEqual(len(dataset_manifest.images), 3)
            self.assertEqual(len(dataset_manifest.labelmap), 3)
            self.assertEqual(dataset_manifest.images[0].id, '0.jpg')
            self.assertEqual(dataset_manifest.images[0].labels, [0, 1])
            self.assertEqual(dataset_manifest.images[1].id, '1.jpg')
            self.assertEqual(dataset_manifest.images[1].labels, [1, 2])
            self.assertEqual(dataset_manifest.images[2].id, '2.jpg')
            self.assertEqual(dataset_manifest.images[2].labels, [2])

    def test_labelmap_exists(self):
        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, 'test.txt')
            with open(file_path, 'w') as f:
                f.write("0.jpg 0,1\n1.jpg 1,2\n2.jpg 2")
            label_file_path = os.path.join(tempdir, 'labels.txt')
            with open(label_file_path, 'w') as f:
                f.write('custom_label0\ncustom_label1\ncustom_label2\n')
            dataset_dict['root_folder'] = str(tempdir)
            dataset_dict['type'] = 'classification_multilabel'
            dataset_dict['labelmap'] = label_file_path
            dataset_manifest = IrisManifestAdaptor.create_dataset_manifest(DatasetInfo(dataset_dict), Usages.TEST_PURPOSE)

            self.assertEqual(dataset_manifest.labelmap, ['custom_label0', 'custom_label1', 'custom_label2'])

    def test_od_manifest(self):
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

            # PIL.Image.new('RGB', (100, 100)).save(os.path.join(tempdir, '0.jpg'))
            # PIL.Image.new('RGB', (100, 100)).save(os.path.join(tempdir, '1.jpg'))

            dataset_manifest = IrisManifestAdaptor.create_dataset_manifest(DatasetInfo(dataset_dict), Usages.TEST_PURPOSE)

            self.assertIsInstance(dataset_manifest, DatasetManifest)
            self.assertEqual(len(dataset_manifest.images), 2)
            self.assertEqual(len(dataset_manifest.labelmap), 4)
            self.assertEqual(dataset_manifest.images[0].labels, [[0, 0.0, 100.0, 0.0, 100.0], [1, 0.0, 100.0, 0.0, 100.0]])
            self.assertEqual(dataset_manifest.images[1].labels, [[1, 50.0, 50.0, 100.0, 100.0], [3, 0.0, 50.0, 100.0, 100.0]])

    def test_od_empty_labels(self):
        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            dataset_dict['root_folder'] = str(tempdir)
            dataset_dict['type'] = 'object_detection'
            with open(os.path.join(tempdir, 'test.txt'), 'w') as f:
                f.write('0.jpg 0.txt\n1.jpg 1.txt')

            with open(os.path.join(tempdir, '0.txt'), 'w'):
                pass

            with open(os.path.join(tempdir, '1.txt'), 'w') as f:
                f.write('1 50 50 100 100\n3 0 50 100 100')

            # PIL.Image.new('RGB', (100, 100)).save(os.path.join(tempdir, '0.jpg'))
            # PIL.Image.new('RGB', (100, 100)).save(os.path.join(tempdir, '1.jpg'))

            dataset_manifest = IrisManifestAdaptor.create_dataset_manifest(DatasetInfo(dataset_dict), Usages.TEST_PURPOSE)

            self.assertIsInstance(dataset_manifest, DatasetManifest)
            self.assertEqual(len(dataset_manifest.images), 2)
            self.assertEqual(len(dataset_manifest.labelmap), 4)
            self.assertEqual(dataset_manifest.images[0].labels, [])
            self.assertEqual(dataset_manifest.images[1].labels, [[1, 50.0, 50.0, 100.0, 100.0], [3, 0.0, 50.0, 100.0, 100.0]])
