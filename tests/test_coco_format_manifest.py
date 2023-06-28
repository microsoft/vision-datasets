import copy
import json
import pathlib
import tempfile
import unittest

import numpy as np
from PIL import Image

from vision_datasets.common import CocoManifestAdaptorFactory, DatasetManifest, DatasetTypes
from vision_datasets.common.data_reader import FileReader
from vision_datasets.image_classification import ImageClassificationLabelManifest
from vision_datasets.multi_task.coco_manifest_adaptor import MultiTaskCocoManifestAdaptor

from .test_dataset_manifest import TestCases, _coco_dict_to_manifest


class TestCreateCocoDatasetManifest(unittest.TestCase):
    def test_file_path_created_right_with_zip_prefix(self):
        image_matting_manifest = {
            "images": [{"id": 1, "file_name": "image/test_1.jpg", "zip_file": "train_images.zip"}],
            "annotations": [
                {"id": 1, "image_id": 1, "label": "mask/test_1.png", 'zip_file': "image_matting_test_data.zip"}
            ]
        }

        with tempfile.TemporaryDirectory() as tempdir:
            file_path = tempdir / pathlib.Path('temp_coco.json')
            file_path.write_text(json.dumps(image_matting_manifest))
            manifest = CocoManifestAdaptorFactory.create(DatasetTypes.IMAGE_MATTING).create_dataset_manifest(str(file_path))

            image = image_matting_manifest['images'][0]
            annotation = image_matting_manifest['annotations'][0]
            self.assertEqual(manifest.images[0].img_path, image['zip_file'] + '@' + image['file_name'])
            self.assertEqual(manifest.images[0].labels[0].label_path, annotation['zip_file'] + '@' + annotation['label'])

    def test_od_respect_iscrowd(self):
        od_manifest = {
            "images": [{"id": 1, "file_name": "image/test_1.jpg", "zip_file": "train_images.zip"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1, "bbox": [10, 10, 80, 80], "iscrowd": 1},
                {"id": 2, "category_id": 1, "image_id": 1, "bbox": [90, 90, 90, 90]},
                {"id": 3, "category_id": 2, "image_id": 1, "bbox": [20, 20, 180, 180]}
            ],
            "categories": [
                {"id": 1, "name": "tiger"},
                {"id": 2, "name": "rabbit"}
            ]
        }

        with tempfile.TemporaryDirectory() as tempdir:
            file_path = tempdir / pathlib.Path('temp_coco2.json')
            file_path.write_text(json.dumps(od_manifest))
            manifest = CocoManifestAdaptorFactory.create(DatasetTypes.IMAGE_OBJECT_DETECTION).create_dataset_manifest(str(file_path))

            image = manifest.images[0]
            self.assertEqual([x.additional_info.get('iscrowd', 0) for x in image.labels], [1, 0, 0])

    def test_image_classification(self):
        dataset_manifest = TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, 0)

        self.assertIsInstance(dataset_manifest, DatasetManifest)
        self.assertEqual(len(dataset_manifest.images), 2)
        self.assertEqual(len(dataset_manifest.categories), 2)
        self.assertEqual([label.category_id for label in dataset_manifest.images[0].labels], [0])
        self.assertEqual([label.category_id for label in dataset_manifest.images[1].labels], [0, 1])

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

        dataset_manifest = _coco_dict_to_manifest(manifest_dict, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)

        self.assertIsInstance(dataset_manifest, DatasetManifest)
        self.assertEqual(len(dataset_manifest.images), 2)
        self.assertEqual(len(dataset_manifest.categories), 2)
        self.assertEqual([label.category_id for label in dataset_manifest.images[0].labels], [0])
        self.assertEqual([label.category_id for label in dataset_manifest.images[1].labels], [0, 1])

    def test_object_detection_bbox_format_LTWH(self):
        dataset_manifest = TestCases.get_manifest(DatasetTypes.IMAGE_OBJECT_DETECTION, 0)

        self.assertIsInstance(dataset_manifest, DatasetManifest)
        self.assertEqual(len(dataset_manifest.images), 2)
        self.assertEqual(len(dataset_manifest.categories), 2)
        self.assertEqual([label.label_data for label in dataset_manifest.images[0].labels], [[0, 10, 10, 100, 100]])
        self.assertEqual([label.label_data for label in dataset_manifest.images[1].labels], [[0, 100, 100, 200, 200], [1, 20, 20, 200, 200]])

    def test_object_detection_bbox_format_LTRB(self):
        manifest_dict = copy.deepcopy(TestCases.od_manifest_dicts[0])
        manifest_dict['bbox_format'] = 'ltrb'

        dataset_manifest = _coco_dict_to_manifest(manifest_dict, DatasetTypes.IMAGE_OBJECT_DETECTION)
        self.assertIsInstance(dataset_manifest, DatasetManifest)
        self.assertEqual(len(dataset_manifest.images), 2)
        self.assertEqual(len(dataset_manifest.categories), 2)
        self.assertEqual([label.label_data for label in dataset_manifest.images[0].labels], [[0, 10, 10, 90, 90]])
        self.assertEqual([label.label_data for label in dataset_manifest.images[1].labels], [[0, 100, 100, 100, 100], [1, 20, 20, 180, 180]])

    def test_image_caption_manifest(self):
        img_0_caption = ['A black Honda motorcycle parked in front of a garage.',
                         'A Honda motorcycle parked in a grass driveway.',
                         'A black Honda motorcycle with a dark burgundy seat.',
                         'Ma motorcycle parked on the gravel in front of a garage.',
                         'A motorcycle with its brake extended standing outside.']
        img_1_caption = ['A picture of a modern looking kitchen area.\n',
                         'A narrow kitchen ending with a chrome refrigerator.',
                         'A narrow kitchen is decorated in shades of white, gray, and black.',
                         'a room that has a stove and a icebox in it',
                         'A long empty, minimal modern skylit home kitchen.']

        dataset_manifest = TestCases.get_manifest(DatasetTypes.IMAGE_CAPTION, 2)
        self.assertIsInstance(dataset_manifest, DatasetManifest)
        self.assertEqual(len(dataset_manifest.images), 2)
        self.assertEqual([label.label_data for label in dataset_manifest.images[0].labels], img_0_caption)
        self.assertEqual([label.label_data for label in dataset_manifest.images[1].labels], img_1_caption)

    def test_multilingual_manifest(self):
        cap_manifests = {
            "images": [{"id": 1, "file_name": "1.jpg", 'zip_file': 'train_images.zip'},
                       {"id": 2, "file_name": "2.jpg", 'zip_file': 'train_images.zip'}],
            "annotations": [
                {"id": 1, "image_id": 1, "caption": "今天天气不错."},
                {"id": 2, "image_id": 2, "caption": "今天天气还可以."},
                {"id": 3, "image_id": 2, "caption": "今日は良い天気."},
                {"id": 4, "image_id": 2, "caption": "Das Wetter ist heute schön."},
            ]
        }

        with tempfile.TemporaryDirectory() as tempdir:
            caption_coco_file_path = pathlib.Path(tempdir) / 'caption_test.json'
            caption_coco_file_path.write_text(json.dumps(cap_manifests))
            manifest = CocoManifestAdaptorFactory.create(DatasetTypes.IMAGE_CAPTION).create_dataset_manifest(caption_coco_file_path)
            caps = [x.label_data for img in manifest.images for x in img.labels]
            assert [x['caption'] for x in cap_manifests['annotations']] == caps

    def test_multitask_ic_multilabel_and_image_caption(self):
        classfication_manifest_dict = TestCases.ic_multilabel_manifest_dicts[0]
        imcap_manifest_dict = TestCases.cap_manifest_dicts[2]
        img_0_caption = ['A black Honda motorcycle parked in front of a garage.',
                         'A Honda motorcycle parked in a grass driveway.',
                         'A black Honda motorcycle with a dark burgundy seat.',
                         'Ma motorcycle parked on the gravel in front of a garage.',
                         'A motorcycle with its brake extended standing outside.']
        img_1_caption = ['A picture of a modern looking kitchen area.\n',
                         'A narrow kitchen ending with a chrome refrigerator.',
                         'A narrow kitchen is decorated in shades of white, gray, and black.',
                         'a room that has a stove and a icebox in it',
                         'A long empty, minimal modern skylit home kitchen.']

        task_types = {'task1': DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, 'task2': DatasetTypes.IMAGE_CAPTION}

        with tempfile.TemporaryDirectory() as tempdir:
            classification_coco_file_path = pathlib.Path(tempdir) / 'classification_test.json'
            classification_coco_file_path.write_text(json.dumps(classfication_manifest_dict))
            imcap_coco_file_path = pathlib.Path(tempdir) / 'imcap_test.json'
            imcap_coco_file_path.write_text(json.dumps(imcap_manifest_dict))

            coco_file_path = {'task1': str(classification_coco_file_path), 'task2': str(imcap_coco_file_path)}
            dataset_manifest = MultiTaskCocoManifestAdaptor(task_types).create_dataset_manifest(coco_file_path)

        self.assertIsInstance(dataset_manifest, DatasetManifest)
        self.assertEqual(len(dataset_manifest.images), 2)
        self.assertEqual(len(dataset_manifest.categories), 2)
        self.assertEqual({x: [z.label_data for z in y] for x, y in dataset_manifest.images[0].labels.items()}, {'task1': [0], 'task2': img_0_caption})
        self.assertEqual({x: [z.label_data for z in y] for x, y in dataset_manifest.images[1].labels.items()}, {'task1': [0, 1], 'task2': img_1_caption})

    def test_image_text_manifest(self):
        for i in range(len(TestCases.image_text_manifest_dicts)):
            dataset_manifest = TestCases.get_manifest(DatasetTypes.IMAGE_TEXT_MATCHING, i)
            self.assertIsInstance(dataset_manifest, DatasetManifest)
            self.assertEqual(len(dataset_manifest.images), len(TestCases.image_text_manifest_dicts[i]['images']))
            self.assertEqual(len([label for image in dataset_manifest.images for label in image.labels]), len(TestCases.image_text_manifest_dicts[i]['annotations']))
            image_ann = {}
            for ann in TestCases.image_text_manifest_dicts[i]['annotations']:
                img_id = ann['image_id']
                image_ann[img_id] = image_ann.get(img_id, [])
                image_ann[img_id].append((ann['text'], ann['match']))
            for image in dataset_manifest.images:
                assert [label.label_data for label in image.labels] == image_ann[image.id]

    def test_image_matting_manifest(self):
        zip_file_path = pathlib.Path(__file__).resolve().parent / 'image_matting_test_data.zip'
        file_reader = FileReader()
        img_0_matting = np.asarray(Image.open(file_reader.open(str(zip_file_path)+'@mask/test_1.png')))
        img_1_matting = np.asarray(Image.open(file_reader.open(str(zip_file_path)+'@mask/test_2.png')))

        dataset_manifest = TestCases.get_manifest(DatasetTypes.IMAGE_MATTING, 2)
        self.assertIsInstance(dataset_manifest, DatasetManifest)
        self.assertEqual(len(dataset_manifest.images), 2)
        self.assertTrue(np.array_equal(dataset_manifest.images[0].labels[0].label_data, img_0_matting, equal_nan=True))
        self.assertTrue(np.array_equal(dataset_manifest.images[1].labels[0].label_data, img_1_matting, equal_nan=True))

    def test_multitask_ic_multilabel_and_image_matting(self):
        classfication_manifest_dict = TestCases.ic_multilabel_manifest_dicts[0]
        image_matting_manifest_dict = TestCases.image_matting_manifest_dicts[2]
        zip_file_path = pathlib.Path(__file__).resolve().parent / 'image_matting_test_data.zip'
        file_reader = FileReader()
        img_0_matting = Image.open(file_reader.open(str(zip_file_path)+'@mask/test_1.png'))
        img_1_matting = Image.open(file_reader.open(str(zip_file_path)+'@mask/test_2.png'))

        task_types = {'task1': DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, 'task2': DatasetTypes.IMAGE_MATTING}

        with tempfile.TemporaryDirectory() as tempdir:
            classification_coco_file_path = pathlib.Path(tempdir) / 'classification_test.json'
            classification_coco_file_path.write_text(json.dumps(classfication_manifest_dict))
            image_matting_coco_file_path = pathlib.Path(tempdir) / 'image_matting_test.json'
            image_matting_coco_file_path.write_text(json.dumps(image_matting_manifest_dict))

            coco_file_path = {'task1': str(classification_coco_file_path), 'task2': str(image_matting_coco_file_path)}
            dataset_manifest = MultiTaskCocoManifestAdaptor(task_types).create_dataset_manifest(coco_file_path)

        self.assertIsInstance(dataset_manifest, DatasetManifest)
        self.assertEqual(len(dataset_manifest.images), 2)
        self.assertEqual(len(dataset_manifest.categories), 2)
        self.assertEqual([x.label_data for x in dataset_manifest.images[0].labels['task1']],  [0])
        self.assertTrue(np.array_equal(dataset_manifest.images[0].labels['task2'][0].label_data, img_0_matting, equal_nan=True))
        self.assertEqual([x.label_data for x in dataset_manifest.images[1].labels['task1']],  [0, 1])
        self.assertTrue(np.array_equal(dataset_manifest.images[1].labels['task2'][0].label_data, img_1_matting, equal_nan=True))

    def test_image_regression_manifest(self):
        image_regression_manifest = {
            "images": [{"id": 1, "file_name": "train_images.zip@1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "target": 1.0},
                {"id": 2, "image_id": 2, "target": 2.0},
            ]
        }

        dataset_manifest = TestCases.get_manifest(DatasetTypes.IMAGE_REGRESSION, 0)
        self.assertIsInstance(dataset_manifest, DatasetManifest)
        self.assertEqual(len(dataset_manifest.images), 2)
        self.assertEqual([label.label_data for label in dataset_manifest.images[0].labels], [image_regression_manifest["annotations"][0]["target"]])
        self.assertEqual([label.label_data for label in dataset_manifest.images[1].labels], [image_regression_manifest["annotations"][1]["target"]])

    def test_classification_dataset_with_bbox(self):
        coco_file = {
            "images": [{"id": 1, "file_name": "image/test_1.jpg", "zip_file": "train_images.zip"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1, "bbox": [10, 10, 80, 80]}
            ],
            "categories": [
                {"id": 1, "name": "tiger"}
            ]
        }

        with tempfile.NamedTemporaryFile() as f:
            pathlib.Path(f.name).write_text(json.dumps(coco_file))
            manifest = CocoManifestAdaptorFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS).create_dataset_manifest(f.name)
            self.assertEqual(manifest.images[0].labels, [ImageClassificationLabelManifest(0, additional_info={'bbox': [10, 10, 80, 80]})])
