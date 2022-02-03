import copy
import json
import os
import pathlib
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

from vision_datasets import IrisManifestAdaptor, DatasetInfo, DatasetManifest, CocoManifestAdaptor
from vision_datasets.common.constants import Usages, DatasetTypes
from vision_datasets.common.data_manifest import ImageDataManifest


def _generate_labelmap(n_classes):
    return [str(i) for i in range(n_classes)]


def _get_instance_count_per_class(manifest):
    if manifest.is_multitask:
        return Counter([manifest._get_cid(label, task_name) for image in manifest.images for task_name, task_labels in image.labels.items() for label in task_labels])
    else:
        return Counter([manifest._get_cid(label) for image in manifest.images for label in image.labels])


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


class TestCreateCocoDatasetManifest(unittest.TestCase):
    DATASET_INFO_DICT = {
        "name": "dummy",
        "version": 1,
        "type": "classification_multiclass",
        "root_folder": "dummy",
        "test": {
            "index_path": "test.json",
            "files_for_local_usage": [
                "Train.zip"
            ]
        },
    }

    def test_image_classification(self):
        manifest_dict = {
            "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "siberian-kitten.jpg"},
                       {"id": 2, "width": 224.0, "height": 224.0, "file_name": "kitten 3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1},
                {"id": 2, "category_id": 1, "image_id": 2},
                {"id": 3, "category_id": 2, "image_id": 2}
            ], "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
        }
        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            dataset_dict['root_folder'] = ''
            dataset_dict['type'] = 'classification_multilabel'
            coco_file_path = os.path.join(tempdir, 'test.json')
            with open(coco_file_path, 'w') as f:
                json.dump(manifest_dict, f)

            dataset_manifest = CocoManifestAdaptor.create_dataset_manifest(coco_file_path, DatasetTypes.IC_MULTILABEL)
            self.assertIsInstance(dataset_manifest, DatasetManifest)
            self.assertEqual(len(dataset_manifest.images), 2)
            self.assertEqual(len(dataset_manifest.labelmap), 2)
            self.assertEqual(dataset_manifest.images[0].labels, [0])
            self.assertEqual(dataset_manifest.images[1].labels, [0, 1])

    def test_index_can_start_from_zero(self):
        manifest_dict = {
            "images": [{"id": 0, "width": 224.0, "height": 224.0, "file_name": "siberian-kitten.jpg"},
                       {"id": 1, "width": 224.0, "height": 224.0, "file_name": "kitten 3.jpg"}],
            "annotations": [
                {"id": 0, "category_id": 0, "image_id": 0},
                {"id": 1, "category_id": 0, "image_id": 1},
                {"id": 2, "category_id": 1, "image_id": 1}
            ], "categories": [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}]
        }
        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            dataset_dict['root_folder'] = ''
            dataset_dict['type'] = 'classification_multilabel'
            coco_file_path = os.path.join(tempdir, 'test.json')
            with open(coco_file_path, 'w') as f:
                json.dump(manifest_dict, f)

            dataset_manifest = CocoManifestAdaptor.create_dataset_manifest(coco_file_path, DatasetTypes.IC_MULTILABEL)
            self.assertIsInstance(dataset_manifest, DatasetManifest)
            self.assertEqual(len(dataset_manifest.images), 2)
            self.assertEqual(len(dataset_manifest.labelmap), 2)
            self.assertEqual(dataset_manifest.images[0].labels, [0])
            self.assertEqual(dataset_manifest.images[1].labels, [0, 1])

    def test_object_detection_bbox_format_LTWH(self):
        manifest_dict = {
            "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "siberian-kitten.jpg"},
                       {"id": 2, "width": 224.0, "height": 224.0, "file_name": "kitten 3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1, "bbox": [10, 10, 90, 90]},
                {"id": 2, "category_id": 1, "image_id": 2, "bbox": [100, 100, 100, 100]},
                {"id": 3, "category_id": 2, "image_id": 2, "bbox": [20, 20, 180, 180]}
            ], "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
        }
        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            dataset_dict['root_folder'] = ''
            dataset_dict['type'] = 'object_detection'
            coco_file_path = os.path.join(tempdir, 'test.json')
            with open(coco_file_path, 'w') as f:
                json.dump(manifest_dict, f)

            dataset_manifest = CocoManifestAdaptor.create_dataset_manifest(coco_file_path, DatasetTypes.OD)
            self.assertIsInstance(dataset_manifest, DatasetManifest)
            self.assertEqual(len(dataset_manifest.images), 2)
            self.assertEqual(len(dataset_manifest.labelmap), 2)
            self.assertEqual(dataset_manifest.images[0].labels, [[0, 10, 10, 100, 100]])
            self.assertEqual(dataset_manifest.images[1].labels, [[0, 100, 100, 200, 200], [1, 20, 20, 200, 200]])

    def test_object_detection_bbox_format_LTRB(self):
        manifest_dict = {
            "bbox_format": "ltrb",
            "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "siberian-kitten.jpg"},
                       {"id": 2, "width": 224.0, "height": 224.0, "file_name": "kitten 3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1, "bbox": [10, 10, 100, 100]},
                {"id": 2, "category_id": 1, "image_id": 2, "bbox": [100, 100, 200, 200]},
                {"id": 3, "category_id": 2, "image_id": 2, "bbox": [20, 20, 200, 200]}
            ], "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
        }
        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            dataset_dict['root_folder'] = ''
            dataset_dict['type'] = 'object_detection'
            coco_file_path = os.path.join(tempdir, 'test.json')
            with open(coco_file_path, 'w') as f:
                json.dump(manifest_dict, f)

            dataset_manifest = CocoManifestAdaptor.create_dataset_manifest(coco_file_path, DatasetTypes.OD)
            self.assertIsInstance(dataset_manifest, DatasetManifest)
            self.assertEqual(len(dataset_manifest.images), 2)
            self.assertEqual(len(dataset_manifest.labelmap), 2)
            self.assertEqual(dataset_manifest.images[0].labels, [[0, 10, 10, 100, 100]])
            self.assertEqual(dataset_manifest.images[1].labels, [[0, 100, 100, 200, 200], [1, 20, 20, 200, 200]])

    def test_image_caption(self):
        manifest_dict = {
            "images": [{"id": 1, "file_name": "train_images.zip@honda.jpg"},
                       {"id": 2, "file_name": "train_images.zip@kitchen.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "caption": "A black Honda motorcycle parked in front of a garage."},
                {"id": 2, "image_id": 1, "caption": "A Honda motorcycle parked in a grass driveway."},
                {"id": 3, "image_id": 1, "caption": "A black Honda motorcycle with a dark burgundy seat."},
                {"id": 4, "image_id": 1, "caption": "Ma motorcycle parked on the gravel in front of a garage."},
                {"id": 5, "image_id": 1, "caption": "A motorcycle with its brake extended standing outside."},
                {"id": 6, "image_id": 2, "caption": "A picture of a modern looking kitchen area.\n"},
                {"id": 7, "image_id": 2, "caption": "A narrow kitchen ending with a chrome refrigerator."},
                {"id": 8, "image_id": 2, "caption": "A narrow kitchen is decorated in shades of white, gray, and black."},
                {"id": 9, "image_id": 2, "caption": "a room that has a stove and a icebox in it"},
                {"id": 10, "image_id": 2, "caption": "A long empty, minimal modern skylit home kitchen."}
            ],
        }
        caption_1 = ['A black Honda motorcycle parked in front of a garage.',
                     'A Honda motorcycle parked in a grass driveway.',
                     'A black Honda motorcycle with a dark burgundy seat.',
                     'Ma motorcycle parked on the gravel in front of a garage.',
                     'A motorcycle with its brake extended standing outside.']
        caption_2 = ['A picture of a modern looking kitchen area.\n',
                     'A narrow kitchen ending with a chrome refrigerator.',
                     'A narrow kitchen is decorated in shades of white, gray, and black.',
                     'a room that has a stove and a icebox in it',
                     'A long empty, minimal modern skylit home kitchen.']

        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            dataset_dict['root_folder'] = ''
            dataset_dict['type'] = 'image_caption'
            coco_file_path = os.path.join(tempdir, 'test.json')
            with open(coco_file_path, 'w') as f:
                json.dump(manifest_dict, f)
            dataset_manifest = CocoManifestAdaptor.create_dataset_manifest(coco_file_path, DatasetTypes.IMCAP)
            self.assertIsInstance(dataset_manifest, DatasetManifest)
            self.assertEqual(len(dataset_manifest.images), 2)
            self.assertEqual(dataset_manifest.images[0].labels, caption_1)
            self.assertEqual(dataset_manifest.images[1].labels, caption_2)

    def test_multitask_ic_multilabel_and_image_caption(self):
        classfication_manifest_dict = {
            "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "siberian-kitten.jpg"},
                       {"id": 2, "width": 224.0, "height": 224.0, "file_name": "kitten 3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1},
                {"id": 2, "category_id": 1, "image_id": 2},
                {"id": 3, "category_id": 2, "image_id": 2}
            ], "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
        }
        imcap_manifest_dict = {
            "images": [{"id": 1, "file_name": "train_images.zip@honda.jpg"},
                       {"id": 2, "file_name": "train_images.zip@kitchen.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "caption": "A black Honda motorcycle parked in front of a garage."},
                {"id": 2, "image_id": 1, "caption": "A Honda motorcycle parked in a grass driveway."},
                {"id": 3, "image_id": 1, "caption": "A black Honda motorcycle with a dark burgundy seat."},
                {"id": 4, "image_id": 1, "caption": "Ma motorcycle parked on the gravel in front of a garage."},
                {"id": 5, "image_id": 1, "caption": "A motorcycle with its brake extended standing outside."},
                {"id": 6, "image_id": 2, "caption": "A picture of a modern looking kitchen area.\n"},
                {"id": 7, "image_id": 2, "caption": "A narrow kitchen ending with a chrome refrigerator."},
                {"id": 8, "image_id": 2, "caption": "A narrow kitchen is decorated in shades of white, gray, and black."},
                {"id": 9, "image_id": 2, "caption": "a room that has a stove and a icebox in it"},
                {"id": 10, "image_id": 2, "caption": "A long empty, minimal modern skylit home kitchen."}
            ],
        }
        caption_1 = ['A black Honda motorcycle parked in front of a garage.',
                     'A Honda motorcycle parked in a grass driveway.',
                     'A black Honda motorcycle with a dark burgundy seat.',
                     'Ma motorcycle parked on the gravel in front of a garage.',
                     'A motorcycle with its brake extended standing outside.']
        caption_2 = ['A picture of a modern looking kitchen area.\n',
                     'A narrow kitchen ending with a chrome refrigerator.',
                     'A narrow kitchen is decorated in shades of white, gray, and black.',
                     'a room that has a stove and a icebox in it',
                     'A long empty, minimal modern skylit home kitchen.']

        dataset_dict = copy.deepcopy(self.DATASET_INFO_DICT)
        with tempfile.TemporaryDirectory() as tempdir:
            dataset_dict['root_folder'] = ''
            dataset_dict['type'] = {'task1': DatasetTypes.IC_MULTILABEL, 'task2': DatasetTypes.IMCAP}
            classification_coco_file_path = os.path.join(tempdir, 'classification_test.json')
            with open(classification_coco_file_path, 'w') as f:
                json.dump(classfication_manifest_dict, f)
            imcap_coco_file_path = os.path.join(tempdir, 'imcap_test.json')
            with open(imcap_coco_file_path, 'w') as f:
                json.dump(imcap_manifest_dict, f)
            coco_file_path = {'task1': classification_coco_file_path, 'task2': imcap_coco_file_path}
            dataset_manifest = CocoManifestAdaptor.create_dataset_manifest(coco_file_path, dataset_dict['type'])
            self.assertIsInstance(dataset_manifest, DatasetManifest)
            self.assertEqual(len(dataset_manifest.images), 2)
            self.assertEqual(len(dataset_manifest.labelmap), 2)
            self.assertEqual(dataset_manifest.images[0].labels, {'task1': [0], 'task2': caption_1})
            self.assertEqual(dataset_manifest.images[1].labels, {'task1': [0, 1], 'task2': caption_2})


class TestManifestFewShotSample(unittest.TestCase):
    def test_multiclass_sample_10_out_of_30(self):
        n_classes = 3
        n_images_per_class = 30
        n_images_sample = 10
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [i]) for i in range(n_classes)] * n_images_per_class
        dataset_manifest = DatasetManifest(images, _generate_labelmap(n_classes), DatasetTypes.IC_MULTICLASS)
        few_shot_manifest = dataset_manifest.sample_few_shot_subset(n_images_sample)
        assert len(few_shot_manifest.images) == n_images_sample * n_classes

        assert _get_instance_count_per_class(few_shot_manifest) == {0: n_images_sample, 1: n_images_sample, 2: n_images_sample}

    def test_multiclass_sample_10_out_of_5(self):
        n_classes = 3
        n_images_per_class = 5
        n_images_sample = 10
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [i]) for i in range(n_classes)] * n_images_per_class
        dataset_manifest = DatasetManifest(images, _generate_labelmap(n_classes), DatasetTypes.IC_MULTICLASS)
        few_shot_manifest = dataset_manifest.sample_few_shot_subset(n_images_sample)
        assert len(few_shot_manifest.images) == n_images_per_class * n_classes

        assert _get_instance_count_per_class(few_shot_manifest) == {0: n_images_per_class, 1: n_images_per_class, 2: n_images_per_class}

    def test_multilabel(self):
        n_classes = 3
        n_images_per_class = 30
        n_images_sample_per_class = 10
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [i, (i + 1) % n_classes]) for i in range(n_classes)] * n_images_per_class
        dataset_manifest = DatasetManifest(images, _generate_labelmap(n_classes), DatasetTypes.IC_MULTILABEL)
        few_shot_manifest = dataset_manifest.sample_few_shot_subset(n_images_sample_per_class)
        assert len(few_shot_manifest.images) == 17
        assert _get_instance_count_per_class(few_shot_manifest) == {0: 11, 1: 10, 2: 13}

    def test_multitask(self):
        n_classes = 3
        n_images_per_class = 30
        n_images_sample_per_class = 10
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, {'a': [i, (i + 1) % n_classes], 'b': [(i + 1) % n_classes, (i + 2) % n_classes]}) for i in range(n_classes)] * n_images_per_class
        dataset_manifest = DatasetManifest(images, {'a': _generate_labelmap(n_classes), 'b': _generate_labelmap(n_classes)},
                                           {'a': DatasetTypes.IC_MULTICLASS, 'b': DatasetTypes.IC_MULTICLASS})
        few_shot_manifest = dataset_manifest.sample_few_shot_subset(n_images_sample_per_class)
        assert len(few_shot_manifest.images) == 17
        assert _get_instance_count_per_class(few_shot_manifest) == {2: 13, 3: 13, 0: 11, 4: 11, 1: 10, 5: 10}


class TestManifestSubsetByRatio(unittest.TestCase):
    def test_multiclass_sample(self):
        num_classes = 10
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [i]) for i in range(num_classes)] * 100
        dataset_manifest = DatasetManifest(images, _generate_labelmap(num_classes), DatasetTypes.IC_MULTICLASS)

        sampled = dataset_manifest.sample_subset_by_ratio(0.5)
        self.assertEqual(len(sampled.images), 500)
        self.assertEqual(_get_instance_count_per_class(sampled), {i: 50 for i in range(num_classes)})

    def test_multilabel(self):
        num_classes = 10

        # All negative images
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, []) for i in range(1000)]
        dataset_manifest = DatasetManifest(images, _generate_labelmap(num_classes), DatasetTypes.IC_MULTILABEL)

        sampled = dataset_manifest.sample_subset_by_ratio(0.5)
        self.assertEqual(len(sampled.images), 500)
        self.assertFalse(_get_instance_count_per_class(sampled))

        # 2 tags per image
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [i, i + 1]) for i in range(num_classes - 1)] * 100
        dataset_manifest = DatasetManifest(images, _generate_labelmap(num_classes), DatasetTypes.IC_MULTILABEL)

        sampled = dataset_manifest.sample_subset_by_ratio(0.5)
        self.assertGreaterEqual(len(sampled.images), 500)
        for n in _get_instance_count_per_class(sampled).values():
            self.assertGreaterEqual(n, 50)

    def test_multitask(self):
        num_classes = 10
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, {'a': [i, i + 1], 'b': [i, i + 1]}) for i in range(num_classes - 1)] * 100
        dataset_manifest = DatasetManifest(images, {'a': _generate_labelmap(num_classes), 'b': _generate_labelmap(num_classes)}, {'a': DatasetTypes.IC_MULTICLASS, 'b': DatasetTypes.IC_MULTICLASS})

        sampled = dataset_manifest.sample_subset_by_ratio(0.5)
        self.assertGreaterEqual(len(sampled.images), 500)
        for n in _get_instance_count_per_class(sampled).values():
            self.assertGreaterEqual(n, 50)

    def test_detection(self):
        num_classes = 10

        # 0 box per image
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, []) for i in range(1000)]
        dataset_manifest = DatasetManifest(images, _generate_labelmap(num_classes), DatasetTypes.OD)

        sampled = dataset_manifest.sample_subset_by_ratio(0.5)
        self.assertEqual(len(sampled.images), 500)
        self.assertFalse(_get_instance_count_per_class(sampled))  # All negative images.

        # 1 box per image
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [[i, 0, 0, 5, 5]]) for i in range(num_classes)] * 100
        dataset_manifest = DatasetManifest(images, _generate_labelmap(num_classes), DatasetTypes.OD)

        sampled = dataset_manifest.sample_subset_by_ratio(0.5)
        self.assertEqual(len(sampled.images), 500)
        self.assertEqual(_get_instance_count_per_class(sampled), {i: 50 for i in range(num_classes)})

        # 2 boxes per image.
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [[i, 0, 0, 5, 5], [i + 1, 0, 0, 5, 5]]) for i in range(num_classes - 1)] * 100
        dataset_manifest = DatasetManifest(images, _generate_labelmap(num_classes), DatasetTypes.OD)

        sampled = dataset_manifest.sample_subset_by_ratio(0.5)
        self.assertGreaterEqual(len(sampled.images), 500)
        for n in _get_instance_count_per_class(sampled).values():
            self.assertGreaterEqual(n, 50)


class TestGreedyFewShotsSampling(unittest.TestCase):
    def test_multiclass_sample(self):
        num_classes = 10
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [i]) for i in range(num_classes)] * 100
        dataset_manifest = DatasetManifest(images, _generate_labelmap(num_classes), DatasetTypes.IC_MULTICLASS)

        sampled = dataset_manifest.sample_few_shots_subset_greedy(1)
        self.assertEqual(len(sampled.images), 10)
        self.assertEqual(_get_instance_count_per_class(sampled), {i: 1 for i in range(num_classes)})

        sampled = dataset_manifest.sample_few_shots_subset_greedy(100)
        self.assertEqual(len(sampled.images), 1000)
        self.assertEqual(_get_instance_count_per_class(sampled), {i: 100 for i in range(num_classes)})

    def test_multilabel(self):
        num_classes = 10

        # All negative images
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, []) for i in range(1000)]
        dataset_manifest = DatasetManifest(images, _generate_labelmap(num_classes), DatasetTypes.IC_MULTILABEL)

        with self.assertRaises(RuntimeError):
            dataset_manifest.sample_few_shots_subset_greedy(10)

        # 2 tags per image
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [i, i + 1]) for i in range(num_classes - 1)] * 100
        dataset_manifest = DatasetManifest(images, _generate_labelmap(num_classes), DatasetTypes.IC_MULTILABEL)

        sampled = dataset_manifest.sample_few_shots_subset_greedy(10)
        self.assertGreaterEqual(len(sampled.images), 50)
        self.assertLessEqual(len(sampled.images), 100)
        for n in _get_instance_count_per_class(sampled).values():
            self.assertGreaterEqual(n, 10)

    def test_multitask(self):
        num_classes = 10
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, {'a': [i, i + 1], 'b': [i, i + 1]}) for i in range(num_classes - 1)] * 100
        dataset_manifest = DatasetManifest(images, {'a': _generate_labelmap(num_classes), 'b': _generate_labelmap(num_classes)}, {'a': DatasetTypes.IC_MULTICLASS, 'b': DatasetTypes.IC_MULTICLASS})

        sampled = dataset_manifest.sample_few_shots_subset_greedy(10)
        self.assertGreaterEqual(len(sampled.images), 50)
        self.assertLessEqual(len(sampled.images), 100)
        for n in _get_instance_count_per_class(sampled).values():
            self.assertGreaterEqual(n, 10)

    def test_detection(self):
        num_classes = 10

        # 0 box per image
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, []) for i in range(1000)]
        dataset_manifest = DatasetManifest(images, _generate_labelmap(num_classes), DatasetTypes.OD)

        with self.assertRaises(RuntimeError):
            dataset_manifest.sample_few_shots_subset_greedy(10)

        # 1 box per image
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [[i, 0, 0, 5, 5]]) for i in range(num_classes)] * 100
        dataset_manifest = DatasetManifest(images, _generate_labelmap(num_classes), DatasetTypes.OD)

        sampled = dataset_manifest.sample_few_shots_subset_greedy(10)
        self.assertEqual(len(sampled.images), 100)
        self.assertEqual(_get_instance_count_per_class(sampled), {i: 10 for i in range(num_classes)})

        # 2 boxes per image.
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [[i, 0, 0, 5, 5], [i + 1, 0, 0, 5, 5]]) for i in range(num_classes - 1)] * 100
        dataset_manifest = DatasetManifest(images, _generate_labelmap(num_classes), DatasetTypes.OD)

        sampled = dataset_manifest.sample_few_shots_subset_greedy(10)
        self.assertGreaterEqual(len(sampled.images), 50)
        self.assertLessEqual(len(sampled.images), 100)
        for n in _get_instance_count_per_class(sampled).values():
            self.assertGreaterEqual(n, 10)

    def test_random_seed(self):
        num_classes = 100
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [i]) for i in range(num_classes)] * 100
        dataset_manifest = DatasetManifest(images, _generate_labelmap(num_classes), DatasetTypes.IC_MULTICLASS)

        for i in range(10):
            sampled = dataset_manifest.sample_few_shots_subset_greedy(1, random_seed=i)
            sampled2 = dataset_manifest.sample_few_shots_subset_greedy(1, random_seed=i)
            self.assertEqual(sampled.images, sampled2.images)


class TestManifestSplit(unittest.TestCase):
    def test_one_image_multiclass(self):
        n_classes = 1
        dataset_manifest = DatasetManifest([ImageDataManifest('1', './1.jpg', 10, 10, [0])], _generate_labelmap(n_classes), DatasetTypes.IC_MULTICLASS)
        train, val = dataset_manifest.train_val_split(1)
        assert len(train.images) == 1
        assert len(val.images) == 0

        dataset_manifest = DatasetManifest([ImageDataManifest('1', './1.jpg', 10, 10, [0])], _generate_labelmap(n_classes), DatasetTypes.IC_MULTICLASS)
        train, val = dataset_manifest.train_val_split(0)
        assert len(train.images) == 0
        assert len(val.images) == 1

    def test_even_multiclass(self):
        n_classes = 3
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [i]) for i in range(n_classes)] * 10
        dataset_manifest = DatasetManifest(images, _generate_labelmap(n_classes), DatasetTypes.IC_MULTICLASS)
        train, val = dataset_manifest.train_val_split(0.70)
        assert len(train.images) == 21
        assert len(val.images) == 9

        assert _get_instance_count_per_class(train) == {0: 7, 1: 7, 2: 7}
        assert _get_instance_count_per_class(val) == {0: 3, 1: 3, 2: 3}

        # test deepcopy
        dataset_copy = copy.deepcopy(dataset_manifest)
        assert dataset_copy

    def test_even_multilabel(self):
        n_classes = 3
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [i, (i + 1) % n_classes]) for i in range(n_classes)] * 10
        dataset_manifest = DatasetManifest(images, _generate_labelmap(n_classes), DatasetTypes.IC_MULTILABEL)
        train, val = dataset_manifest.train_val_split(0.7001)
        assert len(train.images) == 21
        assert len(val.images) == 9

        assert _get_instance_count_per_class(train) == {0: 14, 1: 14, 2: 14}
        assert _get_instance_count_per_class(val) == {0: 6, 1: 6, 2: 6}

        # test deepcopy
        dataset_copy = copy.deepcopy(dataset_manifest)
        assert dataset_copy

    def test_even_detection(self):
        n_classes = 3
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 20, 20, [[i, 0, 0, 10, 10], [(i + 1) % n_classes, 0, 0, 20, 20]]) for i in range(n_classes)] * 10
        dataset_manifest = DatasetManifest(images, _generate_labelmap(n_classes), DatasetTypes.OD)
        train, val = dataset_manifest.train_val_split(0.7001)
        assert len(train.images) == 21
        assert len(val.images) == 9
        assert _get_instance_count_per_class(train) == {0: 14, 1: 14, 2: 14}
        assert _get_instance_count_per_class(val) == {0: 6, 1: 6, 2: 6}

        # test deepcopy
        dataset_copy = copy.deepcopy(dataset_manifest)
        assert dataset_copy

    def test_even_multitask(self):
        n_classes = 3
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, {'a': [i, (i + 1) % n_classes], 'b': [(i + 1) % n_classes, (i + 2) % n_classes]}) for i in range(n_classes)] * 10
        dataset_manifest = DatasetManifest(images, {'a': _generate_labelmap(n_classes), 'b': _generate_labelmap(n_classes)},
                                           {'a': DatasetTypes.IC_MULTICLASS, 'b': DatasetTypes.IC_MULTICLASS})
        train, val = dataset_manifest.train_val_split(0.7001, 3)
        assert len(train.images) == 21
        assert len(val.images) == 9
        assert _get_instance_count_per_class(train) == {0: 14, 1: 14, 2: 14, 3: 14, 4: 14, 5: 14}
        assert _get_instance_count_per_class(val) == {0: 6, 1: 6, 2: 6, 3: 6, 4: 6, 5: 6}

        # test deepcopy
        dataset_copy = copy.deepcopy(dataset_manifest)
        assert dataset_copy


class TestSampleByCategories(unittest.TestCase):
    def test_sample_od_dataset_by_categories(self):
        images = [
            ImageDataManifest(0, './0.jpg', 10, 10, []),
            ImageDataManifest(1, './1.jpg', 10, 10, [[0, 1, 1, 2, 2], [1, 2, 2, 3, 3]]),
            ImageDataManifest(2, './2.jpg', 10, 10, [[1, 1, 1, 2, 2]]),
            ImageDataManifest(3, './3.jpg', 10, 10, [[1, 0, 0, 2, 2], [2, 1, 1, 2, 2], [3, 2, 2, 3, 3]]),
        ]
        manifest = DatasetManifest(images, ['a', 'b', 'c', 'd'], DatasetTypes.OD)
        new_manifest = manifest.sample_categories([1, 3])
        assert len(new_manifest) == len(manifest)
        assert new_manifest.labelmap == ['b', 'd']
        assert new_manifest.images[0].labels == []
        assert new_manifest.images[1].labels == [[0, 2, 2, 3, 3]]
        assert new_manifest.images[2].labels == [[0, 1, 1, 2, 2]]
        assert new_manifest.images[3].labels == [[0, 0, 0, 2, 2], [1, 2, 2, 3, 3]]

    def test_sample_ic_dataset_by_categories(self):
        images = [
            ImageDataManifest(0, './0.jpg', 10, 10, []),
            ImageDataManifest(1, './1.jpg', 10, 10, [0, 1]),
            ImageDataManifest(2, './2.jpg', 10, 10, [1]),
            ImageDataManifest(3, './3.jpg', 10, 10, [1, 2, 3]),
        ]
        manifest = DatasetManifest(images, ['a', 'b', 'c', 'd'], DatasetTypes.IC_MULTILABEL)
        new_manifest = manifest.sample_categories([1, 3])
        assert len(new_manifest) == len(manifest)
        assert new_manifest.labelmap == ['b', 'd']
        assert new_manifest.images[0].labels == []
        assert new_manifest.images[1].labels == [0]
        assert new_manifest.images[2].labels == [0]
        assert new_manifest.images[3].labels == [0, 1]


class TestCocoGeneration(unittest.TestCase):
    def test_coco_generation_od(self):
        manifest_dict = {
            "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "siberian-kitten.jpg"},
                       {"id": 2, "width": 224.0, "height": 224.0, "file_name": "kitten 3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1, "bbox": [10, 10, 90, 90]},
                {"id": 2, "category_id": 1, "image_id": 2, "bbox": [100, 100, 100, 100]},
                {"id": 3, "category_id": 2, "image_id": 2, "bbox": [20, 20, 180, 180]}
            ], "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = pathlib.Path(temp_dir) / 'coco.json'
            temp_file.write_text(json.dumps(manifest_dict))
            manifest = CocoManifestAdaptor.create_dataset_manifest(str(temp_file), DatasetTypes.OD)
            coco_dict = manifest.generate_coco_annotations()

        assert coco_dict == manifest_dict

    def test_coco_generation_ic(self):
        manifest_dict = {
            "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "siberian-kitten.jpg"},
                       {"id": 2, "width": 224.0, "height": 224.0, "file_name": "kitten 3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1},
                {"id": 2, "category_id": 1, "image_id": 2},
                {"id": 3, "category_id": 2, "image_id": 2}
            ], "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = pathlib.Path(temp_dir) / 'coco.json'
            temp_file.write_text(json.dumps(manifest_dict))
            manifest = CocoManifestAdaptor.create_dataset_manifest(str(temp_file), DatasetTypes.IC_MULTICLASS)
            coco_dict = manifest.generate_coco_annotations()

        assert coco_dict == manifest_dict

    def test_coco_generation_caption(self):
        manifest_dict = {
            "images": [{"id": 1, "file_name": "train_images.zip@honda.jpg", "width": 10, "height": 10},
                       {"id": 2, "file_name": "train_images.zip@kitchen.jpg", "width": 10, "height": 10}],
            "annotations": [
                {"id": 1, "image_id": 1, "caption": "A black Honda motorcycle parked in front of a garage."},
                {"id": 2, "image_id": 1, "caption": "A Honda motorcycle parked in a grass driveway."},
                {"id": 3, "image_id": 1, "caption": "A black Honda motorcycle with a dark burgundy seat."},
                {"id": 4, "image_id": 1, "caption": "Ma motorcycle parked on the gravel in front of a garage."},
                {"id": 5, "image_id": 1, "caption": "A motorcycle with its brake extended standing outside."},
                {"id": 6, "image_id": 2, "caption": "A picture of a modern looking kitchen area.\n"},
                {"id": 7, "image_id": 2, "caption": "A narrow kitchen ending with a chrome refrigerator."},
                {"id": 8, "image_id": 2, "caption": "A narrow kitchen is decorated in shades of white, gray, and black."},
                {"id": 9, "image_id": 2, "caption": "a room that has a stove and a icebox in it"},
                {"id": 10, "image_id": 2, "caption": "A long empty, minimal modern skylit home kitchen."}
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = pathlib.Path(temp_dir) / 'coco.json'
            temp_file.write_text(json.dumps(manifest_dict))
            manifest = CocoManifestAdaptor.create_dataset_manifest(str(temp_file), DatasetTypes.IMCAP)
            coco_dict = manifest.generate_coco_annotations()

        assert coco_dict == manifest_dict


if __name__ == '__main__':
    unittest.main()
