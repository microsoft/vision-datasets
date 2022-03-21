import copy
import os
import pathlib
import tempfile
import unittest

from PIL import Image
from vision_datasets import CocoManifestAdaptor, DatasetInfo, ManifestDataset
from vision_datasets.common.constants import Usages, DatasetTypes
from vision_datasets.common.manifest_dataset import DetectionAsClassificationByCroppingDataset, DetectionAsClassificationIgnoreBoxesDataset


class TestDetectionAsClassification(unittest.TestCase):
    DATASET_INFO_DICT = {
        "name": "dummy",
        "version": 1,
        "type": "classification_multiclass",
        "root_folder": "dummy",
        "format": "coco",
        "test": {
            "index_path": "test.txt",
            "files_for_local_usage": [
                "Train.zip"
            ]
        },
    }

    @staticmethod
    def _create_an_od_dataset():
        dataset_dict = copy.deepcopy(TestDetectionAsClassification.DATASET_INFO_DICT)
        tempdir = tempfile.TemporaryDirectory()
        dataset_dict['root_folder'] = tempdir.name
        dataset_dict['type'] = 'object_detection'
        images = [
            {
                'id': i + 1,
                'file_name': f'{i + 1}.jpg',
                'width': 100,
                'height': 100
            } for i in range(2)
        ]

        categories = [
            {
                'id': i + 1,
                'name': f'{i + 1}-class',
            } for i in range(4)
        ]

        annotations = [
            {
                'id': 1,
                'image_id': 1,
                'category_id': 1,
                'bbox': [0, 0, 100, 100]
            },
            {
                'id': 2,
                'image_id': 1,
                'category_id': 2,
                'bbox': [10, 10, 40, 90]
            },
            {
                'id': 3,
                'image_id': 2,
                'category_id': 3,
                'bbox': [50, 50, 30, 30]
            },
            {
                'id': 4,
                'image_id': 2,
                'category_id': 4,
                'bbox': [0, 50, 100, 50]
            }
        ]

        coco_dict = {'images': images, 'categories': categories, 'annotations': annotations}
        for i in range(2):
            Image.new('RGB', (100, 100)).save(os.path.join(tempdir.name, f'{i + 1}.jpg'))

        dataset_info = DatasetInfo(dataset_dict)
        coco_path = pathlib.Path(tempdir.name) / 'coco.json'
        dataset_manifest = CocoManifestAdaptor.create_dataset_manifest(coco_path.name, DatasetTypes.IC_MULTICLASS, tempdir.name)

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

    def test_od_as_ic_dataset_by_crop(self):
        dataset, tempdir = self._create_an_od_dataset()
        with tempdir:
            ic_dataset = DetectionAsClassificationByCroppingDataset(dataset)
            assert ic_dataset.dataset_info.type == DatasetTypes.IC_MULTICLASS, ic_dataset.dataset_info.type
            assert len(ic_dataset) == 4
            assert ic_dataset[0][0].size == (100, 100)
            assert ic_dataset[1][0].size == (40, 90)
            assert ic_dataset[2][0].size == (30, 30)
            assert ic_dataset[3][0].size == (100, 50)

    def test_od_as_ic_dataset_by_ignore_box(self):
        dataset, tempdir = self._create_an_od_dataset()
        with tempdir:
            ic_dataset = DetectionAsClassificationIgnoreBoxesDataset(dataset)
            assert ic_dataset.dataset_info.type == DatasetTypes.IC_MULTILABEL, ic_dataset.dataset_info.type
            assert len(ic_dataset) == 4
            assert ic_dataset[0][0].size == (100, 100)
            assert ic_dataset[1][0].size == (40, 90)
            assert ic_dataset[2][0].size == (30, 30)
            assert ic_dataset[3][0].size == (100, 50)
