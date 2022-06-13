import unittest

from tests.test_fixtures import DetectionTestFixtures
from vision_datasets import ManifestDataset, DatasetInfo
from vision_datasets.common.base_dataset import BaseDataset
from vision_datasets.common.constants import DatasetTypes
from vision_datasets.common.manifest_dataset import DetectionAsClassificationByCroppingDataset, DetectionAsClassificationIgnoreBoxesDataset, LocalFolderCacheDecorator


class TestDetectionAsClassification(unittest.TestCase):
    def test_od_manifest_with_different_coordinate_formats(self):
        dataset, tempdir = DetectionTestFixtures.create_an_od_dataset(2, coordinates='relative')
        with tempdir:
            self.assertEqual(len(dataset), 2)
            self.assertEqual(len(dataset.labels), 4)
            image0, target0, _ = dataset[0]
            image1, target1, _ = dataset[1]
            self.assertEqual(target0, [[0, 0.0, 0.0, 1.0, 1.0], [1, 0.1, 0.1, 0.5, 1.0]])
            self.assertEqual(target1, [[2, 0.5, 0.5, 0.8, 0.8], [3, 0.0, 0.5, 1.0, 1.0]])
            dataset = ManifestDataset(dataset.dataset_info, dataset.dataset_manifest, coordinates='absolute')
            ic_dataset_1 = DetectionAsClassificationByCroppingDataset(dataset)
            image0, target0, _ = dataset[0]
            image1, target1, _ = dataset[1]
            self.assertEqual(target0, [[0, 0.0, 0.0, 100.0, 100.0], [1, 10.0, 10.0, 50.0, 100.0]])
            self.assertEqual(target1, [[2, 50.0, 50.0, 80.0, 80.0], [3, 0.0, 50.0, 100.0, 100.0]])
            ic_dataset_2 = DetectionAsClassificationByCroppingDataset(dataset)
            assert len(ic_dataset_1) == len(ic_dataset_2)
            for i in range(len(ic_dataset_1)):
                img_1 = ic_dataset_1[i]
                img_2 = ic_dataset_2[i]
                assert img_1[0].size == img_2[0].size
                assert img_1[1] == img_2[1]

    def test_od_as_ic_dataset_by_crop(self):
        dataset, tempdir = DetectionTestFixtures.create_an_od_dataset()
        with tempdir:
            ic_dataset = DetectionAsClassificationByCroppingDataset(dataset)
            assert ic_dataset.dataset_info.type == DatasetTypes.IC_MULTICLASS, ic_dataset.dataset_info.type
            assert len(ic_dataset) == 4
            img_1 = ic_dataset[0]
            img_2 = ic_dataset[1]
            img_3 = ic_dataset[2]
            img_4 = ic_dataset[3]
            assert img_1[0].size == (100, 100)
            assert img_1[1] == [0]
            assert img_2[0].size == (40, 90)
            assert img_2[1] == [1]
            assert img_3[0].size == (30, 30)
            assert img_3[1] == [2]
            assert img_4[0].size == (100, 50)
            assert img_4[1] == [3]

    def test_od_as_ic_dataset_by_ignore_box(self):
        dataset, tempdir = DetectionTestFixtures.create_an_od_dataset()
        with tempdir:
            ic_dataset = DetectionAsClassificationIgnoreBoxesDataset(dataset)
            assert ic_dataset.dataset_info.type == DatasetTypes.IC_MULTILABEL, ic_dataset.dataset_info.type
            assert len(ic_dataset) == 2
            img_1 = ic_dataset[0]
            img_2 = ic_dataset[1]
            assert img_1[0].size == (100, 100)
            assert img_1[1] == [0, 1]
            assert img_2[0].size == (100, 100)
            assert img_2[1] == [2, 3]


class TestLocalFolderCacheDecorator(unittest.TestCase):
    class TestDataset(BaseDataset):
        def __len__(self):
            return 0

        def _get_single_item(self, index):
            pass

        def close(self):
            pass

        @property
        def labels(self):
            return []

    def test_respect_dataset_type(self):
        for data_type in set(DatasetTypes.VALID_TYPES) - {DatasetTypes.MULTITASK}:
            cached_dataset = LocalFolderCacheDecorator(TestLocalFolderCacheDecorator.TestDataset(DatasetInfo({'type': data_type, 'name': 'test'})), local_cache_params={'dir': './'})
            assert cached_dataset.dataset_info.type == data_type
            manifest = cached_dataset.generate_manifest()
            assert manifest.data_type == data_type
