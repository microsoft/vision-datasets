import unittest

from tests.test_fixtures import DetectionTestFixtures
from vision_datasets.common.constants import DatasetTypes
from vision_datasets.image_text_matching import VisionAsImageTextDataset
from vision_datasets.image_object_detection import DetectionAsClassificationIgnoreBoxesDataset


class TestVisionAsImageTextDataset(unittest.TestCase):
    def test_od_as_image_text_dataset(self):
        n_images = 3
        dataset, tempdir = DetectionTestFixtures.create_an_od_dataset(n_images)
        with tempdir:
            it_dataset = VisionAsImageTextDataset(dataset)
            assert it_dataset.dataset_info.type == DatasetTypes.IMAGE_TEXT_MATCHING, it_dataset.dataset_info.type
            assert len(it_dataset) == n_images, len(it_dataset)
            matches = [label.label_data[1] for x, labels, _ in it_dataset for label in labels]
            assert len(matches) == n_images * 2
            assert len(set(matches)) == 1 and matches[0] == 1, matches

    def test_od_as_image_text_dataset_with_neg_pairs(self):
        n_images = 3
        n_categories = 10
        dataset, tempdir = DetectionTestFixtures.create_an_od_dataset(n_images, n_categories)
        with tempdir:
            it_dataset = VisionAsImageTextDataset(dataset, 3, rnd_seed=1)
            assert it_dataset.dataset_info.type == DatasetTypes.IMAGE_TEXT_MATCHING, it_dataset.dataset_info.type
            assert len(it_dataset) == n_images, len(it_dataset)
            matches = [label.label_data[1] for x, labels, _ in it_dataset for label in labels]
            assert sum(matches) == 6, matches
            assert len(matches) == 24, len(matches)

    def test_od_as_image_text_dataset_with_down_sampling_neg_pairs(self):
        n_images = 10
        n_categories = 10
        dataset, tempdir = DetectionTestFixtures.create_an_od_dataset(n_images, n_categories)
        with tempdir:
            it_dataset = VisionAsImageTextDataset(dataset, 0.3, rnd_seed=1)
            assert it_dataset.dataset_info.type == DatasetTypes.IMAGE_TEXT_MATCHING, it_dataset.dataset_info.type
            assert len(it_dataset) == n_images, len(it_dataset)
            matches = [label.label_data[1] for x, labels, _ in it_dataset for label in labels]
            assert sum(matches) == 20, matches
            assert len(matches) == 27, matches

    def test_od_as_image_text_dataset_with_neg_pairs_under_expected_ratio(self):
        n_images = 3
        n_categories = 3
        dataset, tempdir = DetectionTestFixtures.create_an_od_dataset(n_images, n_categories)
        with tempdir:
            it_dataset = VisionAsImageTextDataset(dataset, 3, rnd_seed=1)
            assert it_dataset.dataset_info.type == DatasetTypes.IMAGE_TEXT_MATCHING, it_dataset.dataset_info.type
            assert len(it_dataset) == n_images, len(it_dataset)
            matches = [label.label_data[1] for x, labels, _ in it_dataset for label in labels]
            assert sum(matches) == 6, matches
            assert len(matches) == 9, matches


class TestClassificationAsImageTextDataset(unittest.TestCase):
    def test_ic_as_image_text_dataset(self):
        n_images = 10
        n_categories = 10
        dataset, tempdir = DetectionTestFixtures.create_an_od_dataset(n_images, n_categories)
        with tempdir:
            dataset = DetectionAsClassificationIgnoreBoxesDataset(dataset)
            it_dataset = VisionAsImageTextDataset(dataset, 0.3, rnd_seed=1)
            assert it_dataset.dataset_info.type == DatasetTypes.IMAGE_TEXT_MATCHING, it_dataset.dataset_info.type
            assert len(it_dataset) == n_images, len(it_dataset)
            matches = [label.label_data[1] for x, labels, _ in it_dataset for label in labels]
            assert sum(matches) == 20, matches
            assert len(matches) == 27, matches
