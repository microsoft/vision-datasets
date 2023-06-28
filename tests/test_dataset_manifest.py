import copy
import json
import pathlib
import tempfile
import unittest
from collections import Counter

from vision_datasets.common import CategoryManifest, CocoDictGeneratorFactory, CocoManifestAdaptorFactory, DatasetFilter, DatasetManifest, DatasetTypes, ImageDataManifest, ImageNoAnnotationFilter, \
    ManifestMerger, ManifestMergeStrategyFactory, ManifestSampler, RemoveCategories, RemoveCategoriesConfig, SampleByFewShotConfig, SampleByNumSamplesConfig, \
    SampleStrategyFactory, SampleStrategyType, SpawnConfig, SpawnFactory, SplitConfig, SplitFactory
from vision_datasets.common.data_manifest.utils import generate_multitask_dataset_manifest
from vision_datasets.image_classification.manifest import ImageClassificationLabelManifest
from vision_datasets.image_object_detection.manifest import ImageObjectDetectionLabelManifest


def _generate_categories(n_classes):
    return [CategoryManifest(i, str(i)) for i in range(n_classes)]


def _get_instance_count_per_class(manifest: DatasetManifest):
    assert not manifest.is_multitask
    return Counter([label.category_id for image in manifest.images for label in image.labels])


def _coco_dict_to_manifest(coco_dict, data_type):
    with tempfile.TemporaryDirectory() as temp_dir:
        dm1_path = pathlib.Path(temp_dir) / 'coco.json'
        dm1_path.write_text(json.dumps(coco_dict))
        return CocoManifestAdaptorFactory.create(data_type).create_dataset_manifest(str(dm1_path))


class TestCases:
    ic_multiclass_manifest_dicts = [
        {
            "images": [
                {"id": 1, "width": 224.0, "height": 224.0, "file_name": "train/1.jpg"},
                {"id": 2, "width": 224.0, "height": 224.0, "file_name": "train/3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1},
                {"id": 2, "category_id": 2, "image_id": 2}
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        },
        {
            "images": [
                {"id": 1, "width": 224.0, "height": 224.0, "file_name": "test/1.jpg"},
                {"id": 2, "width": 224.0, "height": 224.0, "file_name": "test/2.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1},
                {"id": 2, "category_id": 2, "image_id": 2}
            ],
            "categories": [
                {"id": 1, "name": "tiger"},
                {"id": 2, "name": "rabbit"}
            ]
        },
        {
            "images": [
                {"id": 1, "width": 224.0, "height": 224.0, "file_name": "test/1.jpg"},
                {"id": 2, "width": 224.0, "height": 224.0, "file_name": "test/2.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1},
                {"id": 2, "category_id": 2, "image_id": 2}
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        }]

    ic_multilabel_manifest_dicts = [
        {
            "images": [
                {"id": 1, "width": 224.0, "height": 224.0, "file_name": "train/1.jpg"},
                {"id": 2, "width": 224.0, "height": 224.0, "file_name": "train/3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1},
                {"id": 2, "category_id": 1, "image_id": 2},
                {"id": 3, "category_id": 2, "image_id": 2}
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        },
        {
            "images": [
                {"id": 1, "width": 224.0, "height": 224.0, "file_name": "test/1.jpg"},
                {"id": 2, "width": 224.0, "height": 224.0, "file_name": "test/2.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1},
                {"id": 2, "category_id": 1, "image_id": 2},
                {"id": 3, "category_id": 2, "image_id": 2}
            ],
            "categories": [
                {"id": 1, "name": "tiger"},
                {"id": 2, "name": "rabbit"}
            ]
        }]

    od_manifest_dicts = [
        {
            "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "siberian-kitten.jpg"},
                       {"id": 2, "width": 224.0, "height": 224.0, "file_name": "kitten 3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1, "bbox": [10, 10, 90, 90]},
                {"id": 2, "category_id": 1, "image_id": 2, "bbox": [100, 100, 100, 100]},
                {"id": 3, "category_id": 2, "image_id": 2, "bbox": [20, 20, 180, 180]}
            ],
            "categories": [
                {"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}
            ]
        },
        {
            "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "siberian-kitten.jpg"},
                       {"id": 2, "width": 224.0, "height": 224.0, "file_name": "kitten 3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1, "bbox": [10, 10, 80, 80]},
                {"id": 2, "category_id": 1, "image_id": 2, "bbox": [90, 90, 90, 90]},
                {"id": 3, "category_id": 2, "image_id": 2, "bbox": [20, 20, 180, 180]}
            ],
            "categories": [
                {"id": 1, "name": "tiger"}, {"id": 2, "name": "rabbit"}
            ]
        },
        {
            "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "siberian-kitten.jpg"},
                       {"id": 2, "width": 224.0, "height": 224.0, "file_name": "kitten 3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1, "bbox": [10, 10, 80, 80]},
                {"id": 2, "category_id": 2, "image_id": 2, "bbox": [90, 90, 90, 90]},
            ],
            "categories": [
                {"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}
            ]
        }]

    cap_manifest_dicts = [
        {
            "images": [{"id": 1, "file_name": "train_images.zip@1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "caption": "test 1."},
                {"id": 2, "image_id": 2, "caption": "test 2."},
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@3.jpg"},
                       {"id": 2, "file_name": "train_images.zip@4.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "caption": "test 3."},
                {"id": 2, "image_id": 2, "caption": "test 4."},
            ]
        },
        {
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
        }]

    image_text_manifest_dicts = [
        {
            "images": [{"id": 1, "file_name": "train_images.zip@1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "text": "test 1.", "match": 0},
                {"id": 2, "image_id": 2, "text": "test 2.", "match": 0},
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@3.jpg"},
                       {"id": 2, "file_name": "train_images.zip@4.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "text": "test 3.", "match": 0},
                {"id": 2, "image_id": 2, "text": "test 4.", "match": 1},
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@honda.jpg"},
                       {"id": 2, "file_name": "train_images.zip@kitchen.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "text": "A black Honda motorcycle parked in front of a garage.", 'match': 0},
                {"id": 2, "image_id": 1, "text": "A Honda motorcycle parked in a grass driveway.", 'match': 1},
                {"id": 3, "image_id": 1, "text": "A black Honda motorcycle with a dark burgundy seat.", 'match': 1},
                {"id": 4, "image_id": 1, "text": "Ma motorcycle parked on the gravel in front of a garage.", 'match': 0},
                {"id": 5, "image_id": 1, "text": "A motorcycle with its brake extended standing outside.", 'match': 0},
                {"id": 6, "image_id": 2, "text": "A picture of a modern looking kitchen area.\n", 'match': 1},
                {"id": 7, "image_id": 2, "text": "A narrow kitchen ending with a chrome refrigerator.", 'match': 0},
                {"id": 8, "image_id": 2, "text": "A narrow kitchen is decorated in shades of white, gray, and black.", 'match': 0},
                {"id": 9, "image_id": 2, "text": "a room that has a stove and a icebox in it", 'match': 0},
                {"id": 10, "image_id": 2, "text": "A long empty, minimal modern skylit home kitchen.", 'match': 1}
            ],
        }]

    image_matting_manifest_dicts = [
        {
            "images": [{"id": 1, "file_name": "train_images.zip@image/test_1.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "label": f"{str(pathlib.Path(__file__).resolve().parent)}/image_matting_test_data.zip@mask/test_1.png"}
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@image/test_1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@image/test_2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "label": f"{str(pathlib.Path(__file__).resolve().parent)}/image_matting_test_data.zip@mask/test_1.png"},
                {"id": 2, "image_id": 2, "label": f"{str(pathlib.Path(__file__).resolve().parent)}/image_matting_test_data.zip@mask/test_2.png"},
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@image/test_1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@image/test_2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "label": f"{str(pathlib.Path(__file__).resolve().parent)}/image_matting_test_data.zip@mask/test_1.png"},
                {"id": 2, "image_id": 2, "label": f"{str(pathlib.Path(__file__).resolve().parent)}/image_matting_test_data.zip@mask/test_2.png"},
            ]
        }]

    image_regression_manifest_dicts = [
        {
            "images": [{"id": 1, "file_name": "train_images.zip@1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "target": 1.0},
                {"id": 2, "image_id": 2, "target": 2.0},
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@3.jpg"},
                       {"id": 2, "file_name": "train_images.zip@4.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "target": 3.0},
                {"id": 2, "image_id": 2, "target": 4.0},
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@3.jpg"},
                       {"id": 2, "file_name": "train_images.zip@4.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "target": 5.0},
                {"id": 2, "image_id": 2, "target": 6.0},
            ],
        }]

    image_retrieval_dicts = [
        {
            "images": [
                {"id": 1, "file_name": "test1.zip@test/0/image_1.jpg"}, {"id": 2, "file_name": "test2.zip@test/1/image_2.jpg"}
            ],
            "annotations": [
                {"image_id": 1, "id": 1, "query": "men giving a speech"},
                {"image_id": 2, "id": 2, "query": "men not giving a speech"}
            ]
        },
        {
            "images": [
                {"id": 1, "file_name": "test1.zip@test/0/image_1.jpg"}, {"id": 2, "file_name": "test2.zip@test/1/image_2.jpg"}
            ],
            "annotations": [
                {"image_id": 1, "id": 1, "query": "women giving a speech"},
                {"image_id": 2, "id": 2, "query": "women not giving a speech"}
            ]
        }
    ]

    manifest_dict_by_data_type = {
        DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL: ic_multilabel_manifest_dicts,
        DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS: ic_multiclass_manifest_dicts,
        DatasetTypes.IMAGE_OBJECT_DETECTION: od_manifest_dicts,
        DatasetTypes.IMAGE_CAPTION: cap_manifest_dicts,
        DatasetTypes.IMAGE_TEXT_MATCHING: image_text_manifest_dicts,
        DatasetTypes.IMAGE_MATTING: image_matting_manifest_dicts,
        DatasetTypes.IMAGE_REGRESSION: image_regression_manifest_dicts,
        DatasetTypes.TEXT_2_IMAGE_RETRIEVAL: image_retrieval_dicts
    }

    @staticmethod
    def get_manifest(data_type, index):
        return _coco_dict_to_manifest(TestCases.manifest_dict_by_data_type[data_type][index], data_type)


class TestManifestRemoveImagesWithNoLabel(unittest.TestCase):
    def test_detection(self):
        num_classes = 10
        filter = DatasetFilter(ImageNoAnnotationFilter())
        # 0 box per image
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, []) for i in range(1000)]
        manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_OBJECT_DETECTION)

        sampled = filter.run(manifest)
        self.assertEqual(len(sampled.images), 0)
        self.assertFalse(_get_instance_count_per_class(sampled))  # All negative images.

        # 1 box per image
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [ImageObjectDetectionLabelManifest([i, 0, 0, 5, 5])])
                  for i in range(num_classes)] * 100 + [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, []) for i in range(100)]
        manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_OBJECT_DETECTION)

        sampled = filter.run(manifest)
        self.assertEqual(len(sampled.images), 1000)


class TestSampleManifestByNumSamples(unittest.TestCase):
    def test_multiclass_sample(self):
        num_classes = 10
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [ImageClassificationLabelManifest(i)]) for i in range(num_classes)] * 100
        manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)

        strategy = SampleStrategyFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, SampleStrategyType.NumSamples, SampleByNumSamplesConfig(0, False, 500))
        sampler = ManifestSampler(strategy)
        sampled = sampler.run(manifest)
        self.assertEqual(len(sampled.images), 500)

    def test_multilabel(self):
        num_classes = 10

        # All negative images
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, []) for i in range(1000)]
        manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)

        strategy = SampleStrategyFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, SampleStrategyType.NumSamples, SampleByNumSamplesConfig(0, False, 500))
        sampler = ManifestSampler(strategy)
        sampled = sampler.run(manifest)
        self.assertEqual(len(sampled.images), 500)
        self.assertFalse(_get_instance_count_per_class(sampled))

        # 2 tags per image
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [ImageClassificationLabelManifest(i), ImageClassificationLabelManifest(i + 1)]) for i in range(num_classes - 1)] * 100
        manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)

        sampled = sampler.run(manifest)
        self.assertGreaterEqual(len(sampled.images), 500)
        for n in _get_instance_count_per_class(sampled).values():
            self.assertGreaterEqual(n, 50)

    # def test_multitask(self):
    #     num_classes = 10
    #     images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, {'a': [i, i + 1], 'b': [i, i + 1]}) for i in range(num_classes - 1)] * 100
    #     manifest = DatasetManifest(images, {'a': _generate_categories(num_classes), 'b': _generate_categories(num_classes)}, {
    #         'a': DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, 'b': DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS})
    #     strategy = SampleStrategyFactory.create(DatasetTypes.MULTITASK, SampleStrategyType.NumSamples, SampleByNumSamplesConfig(0, False, 500))
    #     sampler = ManifestSampler(strategy)
    #     sampled = sampler.run(manifest)
    #     self.assertGreaterEqual(len(sampled.images), 500)
    #     for n in _get_instance_count_per_class(sampled).values():
    #         self.assertGreaterEqual(n, 50)

    def test_detection(self):
        num_classes = 10

        # 0 box per image
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, []) for i in range(1000)]
        manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_OBJECT_DETECTION)
        strategy = SampleStrategyFactory.create(DatasetTypes.IMAGE_OBJECT_DETECTION, SampleStrategyType.NumSamples, SampleByNumSamplesConfig(0, False, 500))
        sampler = ManifestSampler(strategy)
        sampled = sampler.run(manifest)
        self.assertEqual(len(sampled.images), 500)
        self.assertFalse(_get_instance_count_per_class(sampled))  # All negative images.

        # 1 box per image
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [ImageObjectDetectionLabelManifest([i, 0, 0, 5, 5])]) for i in range(num_classes)] * 100
        manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_OBJECT_DETECTION)

        sampled = sampler.run(manifest)
        self.assertEqual(len(sampled.images), 500)

        # 2 boxes per image.
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [ImageObjectDetectionLabelManifest([i, 0, 0, 5, 5]),
                                    ImageObjectDetectionLabelManifest([i + 1, 0, 0, 5, 5])]) for i in range(num_classes - 1)] * 100
        manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_OBJECT_DETECTION)

        sampled = sampler.run(manifest)
        self.assertGreaterEqual(len(sampled.images), 500)
        for n in _get_instance_count_per_class(sampled).values():
            self.assertGreaterEqual(n, 50)


class TestGreedyFewShotsSampling(unittest.TestCase):
    def test_multiclass_sample(self):
        num_classes = 10
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [ImageClassificationLabelManifest(i)]) for i in range(num_classes)] * 100
        dataset_manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)

        strategy = SampleStrategyFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, SampleStrategyType.FewShot, SampleByFewShotConfig(0, 1))
        sampler = ManifestSampler(strategy)
        sampled = sampler.run(dataset_manifest)
        self.assertEqual(len(sampled.images), 10)
        self.assertEqual(_get_instance_count_per_class(sampled), {i: 1 for i in range(num_classes)})

        strategy = SampleStrategyFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, SampleStrategyType.FewShot, SampleByFewShotConfig(0, 100))
        sampler = ManifestSampler(strategy)
        sampled = sampler.run(dataset_manifest)
        self.assertEqual(len(sampled.images), 1000)
        self.assertEqual(_get_instance_count_per_class(sampled), {i: 100 for i in range(num_classes)})

    def test_multilabel(self):
        num_classes = 10

        # All negative images
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, []) for i in range(1000)]
        dataset_manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)

        with self.assertRaises(RuntimeError):
            strategy = SampleStrategyFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, SampleStrategyType.FewShot, SampleByFewShotConfig(0, 10))
            sampler = ManifestSampler(strategy)
            sampled = sampler.run(dataset_manifest)

        # 2 tags per image
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [ImageClassificationLabelManifest(i), ImageClassificationLabelManifest(i + 1)]) for i in range(num_classes - 1)] * 100
        dataset_manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)

        sampled = sampler.run(dataset_manifest)
        self.assertGreaterEqual(len(sampled), 50)
        self.assertLessEqual(len(sampled), 100)
        for n in _get_instance_count_per_class(sampled).values():
            self.assertGreaterEqual(n, 10)

    # def test_multitask(self):
    #     num_classes = 10
    #     images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, {
    #         'a': [ImageClassificationLabelManifest(i), ImageClassificationLabelManifest(i + 1)],
    #         'b': [ImageClassificationLabelManifest(i), ImageClassificationLabelManifest(i + 1)]
    #     }) for i in range(num_classes - 1)] * 100
    #     dataset_manifest = DatasetManifest(images,
    #                                        {'a': _generate_categories(num_classes), 'b': _generate_categories(num_classes)},
    #                                        {'a': DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, 'b': DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS})

    #     strategy = SampleStrategyFactory.create(DatasetTypes.MULTITASK, SampleStrategyType.FewShot, SampleByFewShotConfig(0, 10))
    #     sampler = ManifestSampler(strategy)
    #     sampled = sampler.run(dataset_manifest)
    #     self.assertGreaterEqual(len(sampled), 50)
    #     self.assertLessEqual(len(sampled), 100)
    #     for n in _get_instance_count_per_class(sampled).values():
    #         self.assertGreaterEqual(n, 10)

    def test_detection(self):
        num_classes = 10

        # 0 box per image
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, []) for i in range(1000)]
        dataset_manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_OBJECT_DETECTION)

        with self.assertRaises(RuntimeError):
            strategy = SampleStrategyFactory.create(DatasetTypes.IMAGE_OBJECT_DETECTION, SampleStrategyType.FewShot, SampleByFewShotConfig(0, 10))
            sampler = ManifestSampler(strategy)
            sampled = sampler.run(dataset_manifest)

        # 1 box per image
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [ImageObjectDetectionLabelManifest([i, 0, 0, 5, 5])]) for i in range(num_classes)] * 100
        dataset_manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_OBJECT_DETECTION)

        sampled = sampler.run(dataset_manifest)
        self.assertEqual(len(sampled.images), 100)
        self.assertEqual(_get_instance_count_per_class(sampled), {i: 10 for i in range(num_classes)})

        # 2 boxes per image.
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [ImageObjectDetectionLabelManifest([i, 0, 0, 5, 5]),
                                    ImageObjectDetectionLabelManifest([i + 1, 0, 0, 5, 5])]) for i in range(num_classes - 1)] * 100
        dataset_manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_OBJECT_DETECTION)

        sampled = sampler.run(dataset_manifest)
        self.assertGreaterEqual(len(sampled.images), 50)
        self.assertLessEqual(len(sampled.images), 100)
        for n in _get_instance_count_per_class(sampled).values():
            self.assertGreaterEqual(n, 10)

    def test_consistency_random_seed(self):
        num_classes = 100
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [ImageClassificationLabelManifest(i)]) for i in range(num_classes)] * 100
        dataset_manifest = DatasetManifest(images, _generate_categories(num_classes), DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)

        for i in range(10):
            n_sample_per_class = 1
            strategy = SampleStrategyFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, SampleStrategyType.FewShot, SampleByFewShotConfig(i, n_sample_per_class))
            sampler = ManifestSampler(strategy)
            sampled = sampler.run(dataset_manifest)
            sampled2 = sampler.run(dataset_manifest)
            self.assertEqual(len(sampled), 100)
            self.assertEqual(len(sampled), len(sampled2))
            self.assertEqual([x.id for x in sampled.images], [x.id for x in sampled2.images])


class TestManifestSplit(unittest.TestCase):
    def test_one_image_multiclass(self):
        n_classes = 1
        dataset_manifest = DatasetManifest([ImageDataManifest('1', './1.jpg', 10, 10, [ImageClassificationLabelManifest(0)])],
                                           _generate_categories(n_classes), DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)
        splitter = SplitFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, SplitConfig(1))
        first, second = splitter.run(dataset_manifest)
        assert len(first.images) == 1
        assert len(second.images) == 0

        splitter = SplitFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, SplitConfig(0))
        first, second = splitter.run(dataset_manifest)
        assert len(first.images) == 0
        assert len(second.images) == 1

    def test_even_multiclass(self):
        n_classes = 3
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [ImageClassificationLabelManifest(i)]) for i in range(n_classes)] * 10
        dataset_manifest = DatasetManifest(images, _generate_categories(n_classes), DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)
        splitter = SplitFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, SplitConfig(0.7))
        first, second = splitter.run(dataset_manifest)
        assert len(first.images) == 21
        assert len(second.images) == 9

        assert _get_instance_count_per_class(first) == {0: 7, 1: 7, 2: 7}
        assert _get_instance_count_per_class(second) == {0: 3, 1: 3, 2: 3}

        # test deepcopy
        dataset_copy = copy.deepcopy(dataset_manifest)
        assert dataset_copy

    def test_even_multilabel(self):
        n_classes = 3
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 10, 10, [ImageClassificationLabelManifest(i), ImageClassificationLabelManifest((i + 1) % n_classes)]) for i in range(n_classes)] * 10
        dataset_manifest = DatasetManifest(images, _generate_categories(n_classes), DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)
        splitter = SplitFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, SplitConfig(0.7001))
        first, second = splitter.run(dataset_manifest)
        assert len(first.images) == 21
        assert len(second.images) == 9

        assert _get_instance_count_per_class(first) == {0: 14, 1: 14, 2: 14}
        assert _get_instance_count_per_class(second) == {0: 6, 1: 6, 2: 6}

        # test deepcopy
        dataset_copy = copy.deepcopy(dataset_manifest)
        assert dataset_copy

    def test_even_detection(self):
        n_classes = 3
        images = [ImageDataManifest(f'{i}', f'./{i}.jpg', 20, 20, [ImageObjectDetectionLabelManifest([i, 0, 0, 10, 10]),
                                    ImageObjectDetectionLabelManifest([(i + 1) % n_classes, 0, 0, 20, 20])]) for i in range(n_classes)] * 10
        dataset_manifest = DatasetManifest(images, _generate_categories(n_classes), DatasetTypes.IMAGE_OBJECT_DETECTION)
        splitter = SplitFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, SplitConfig(0.7001))
        first, second = splitter.run(dataset_manifest)
        assert len(first.images) == 21
        assert len(second.images) == 9
        assert _get_instance_count_per_class(first) == {0: 14, 1: 14, 2: 14}
        assert _get_instance_count_per_class(second) == {0: 6, 1: 6, 2: 6}

        # test deepcopy
        dataset_copy = copy.deepcopy(dataset_manifest)
        assert dataset_copy

    # def test_even_multitask(self):
    #     n_classes = 3
    #     images = [ImageDataManifest(
    #         str(i),
    #         f'./{i}.jpg',
    #         10,
    #         10,
    #         {'a': [ImageClassificationLabelManifest(i), ImageClassificationLabelManifest((i + 1) % n_classes)],
    #             'b': [ImageClassificationLabelManifest((i + 1) % n_classes), ImageClassificationLabelManifest((i + 2) % n_classes)]}) for i in range(n_classes)] * 10

    #     dataset_manifest = DatasetManifest(images, {'a': _generate_categories(n_classes), 'b': _generate_categories(n_classes)},
    #                                        {'a': DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, 'b': DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS})
    #     splitter = SplitFactory.create(DatasetTypes.MULTITASK, SplitConfig(0.7001))
    #     first, second = splitter.run(dataset_manifest)
    #     assert len(first.images) == 21
    #     assert len(second.images) == 9
    #     assert _get_instance_count_per_class(first) == {0: 14, 1: 14, 2: 14, 3: 14, 4: 14, 5: 14}
    #     assert _get_instance_count_per_class(second) == {0: 6, 1: 6, 2: 6, 3: 6, 4: 6, 5: 6}

    #     # test deepcopy
    #     dataset_copy = copy.deepcopy(dataset_manifest)
    #     assert dataset_copy


class TestSampleByCategories(unittest.TestCase):

    def test_sample_od_dataset_by_categories(self):
        images = [
            ImageDataManifest(0, './0.jpg', 10, 10, []),
            ImageDataManifest(1, './1.jpg', 10, 10, [ImageObjectDetectionLabelManifest([0, 1, 1, 2, 2]), ImageObjectDetectionLabelManifest([1, 2, 2, 3, 3])]),
            ImageDataManifest(2, './2.jpg', 10, 10, [ImageObjectDetectionLabelManifest([1, 1, 1, 2, 2])]),
            ImageDataManifest(3, './3.jpg', 10, 10, [ImageObjectDetectionLabelManifest([1, 0, 0, 2, 2]),
                              ImageObjectDetectionLabelManifest([2, 1, 1, 2, 2]), ImageObjectDetectionLabelManifest([3, 2, 2, 3, 3])]),
        ]
        manifest = DatasetManifest(images, [CategoryManifest(i, x) for i, x in enumerate(['a', 'b', 'c', 'd'])], DatasetTypes.IMAGE_OBJECT_DETECTION)
        remover = RemoveCategories(RemoveCategoriesConfig(['a', 'c']))
        new_manifest = remover.run(manifest)
        assert len(new_manifest) == len(manifest)
        assert [x.name for x in new_manifest.categories] == ['b', 'd']
        assert new_manifest.images[0].labels == []
        assert [x.label_data for x in new_manifest.images[1].labels] == [[0, 2, 2, 3, 3]]
        assert [x.label_data for x in new_manifest.images[2].labels] == [[0, 1, 1, 2, 2]]
        assert [x.label_data for x in new_manifest.images[3].labels] == [[0, 0, 0, 2, 2], [1, 2, 2, 3, 3]]

    def test_sample_ic_dataset_by_categories(self):
        images = [
            ImageDataManifest(0, './0.jpg', 10, 10, []),
            ImageDataManifest(1, './1.jpg', 10, 10, [ImageClassificationLabelManifest(0), ImageClassificationLabelManifest(1)]),
            ImageDataManifest(2, './2.jpg', 10, 10, [ImageClassificationLabelManifest(1)]),
            ImageDataManifest(3, './3.jpg', 10, 10, [ImageClassificationLabelManifest(1), ImageClassificationLabelManifest(2), ImageClassificationLabelManifest(3)]),
        ]
        manifest = DatasetManifest(images, [CategoryManifest(i, x) for i, x in enumerate(['a', 'b', 'c', 'd'])], DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)
        remover = RemoveCategories(RemoveCategoriesConfig(['a', 'c']))
        new_manifest = remover.run(manifest)
        assert len(new_manifest) == len(manifest)
        assert [x.name for x in new_manifest.categories] == ['b', 'd']
        assert new_manifest.images[0].labels == []
        assert [x.label_data for x in new_manifest.images[1].labels] == [0]
        assert [x.label_data for x in new_manifest.images[2].labels] == [0]
        assert [x.label_data for x in new_manifest.images[3].labels] == [0, 1]


class TestSpawn(unittest.TestCase):
    def test_spawn_od_manifest(self):
        images = [
            ImageDataManifest(0, './0.jpg', 10, 10, []),
            ImageDataManifest(1, './1.jpg', 10, 10, [ImageObjectDetectionLabelManifest([0, 1, 1, 2, 2]), ImageObjectDetectionLabelManifest([1, 2, 2, 3, 3])]),
            ImageDataManifest(2, './2.jpg', 10, 10, [ImageObjectDetectionLabelManifest([1, 1, 1, 2, 2])]),
            ImageDataManifest(3, './3.jpg', 10, 10, [ImageObjectDetectionLabelManifest([1, 0, 0, 2, 2]),
                              ImageObjectDetectionLabelManifest([2, 1, 1, 2, 2]), ImageObjectDetectionLabelManifest([3, 2, 2, 3, 3])]),
        ]
        dst_size = 120
        manifest = DatasetManifest(images, [CategoryManifest(i, x) for i, x in enumerate(['a', 'b', 'c', 'd'])], DatasetTypes.IMAGE_OBJECT_DETECTION)
        spawner = SpawnFactory.create(DatasetTypes.IMAGE_OBJECT_DETECTION, SpawnConfig(0, dst_size))
        new_manifest = spawner.run(manifest)
        self.assertEqual(len(new_manifest), dst_size)

        spawner = SpawnFactory.create(DatasetTypes.IMAGE_OBJECT_DETECTION, SpawnConfig(0, dst_size, [0., 0.5, 0.5, 1.]))
        new_manifest = spawner.run(manifest)
        cnt = self.cnt_multiclass_labels(new_manifest)
        self.assertEqual(cnt, [30, 120, 60, 60])

    def test_spawn_ic_multilabel_manifest(self):
        images = [
            ImageDataManifest(0, './0.jpg', 10, 10, []),
            ImageDataManifest(1, './1.jpg', 10, 10, [ImageClassificationLabelManifest(0), ImageClassificationLabelManifest(1)]),
            ImageDataManifest(1, './1.jpg', 10, 10, [ImageClassificationLabelManifest(0), ImageClassificationLabelManifest(1)]),
            ImageDataManifest(2, './2.jpg', 10, 10, [ImageClassificationLabelManifest(1)]),
        ]
        dst_size = 60
        manifest = DatasetManifest(images, [CategoryManifest(0, 'a'), CategoryManifest(1, 'b')], DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)
        spawner = SpawnFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, SpawnConfig(0, dst_size))
        new_manifest = spawner.run(manifest)
        self.assertEqual(len(new_manifest), dst_size)
        spawner = SpawnFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, SpawnConfig(0, dst_size, [20, 10, 10, 20]))
        new_manifest = spawner.run(manifest)
        cnt = self.cnt_multiclass_labels(new_manifest)
        self.assertEqual(cnt, [20, 40])

    def test_spawn_ic_multiclass_manifest(self):
        images = [
            ImageDataManifest(0, './0.jpg', 10, 10, [ImageClassificationLabelManifest(0)]),
            ImageDataManifest(1, './1.jpg', 10, 10, [ImageClassificationLabelManifest(0)]),
            ImageDataManifest(1, './1.jpg', 10, 10, [ImageClassificationLabelManifest(2)]),
            ImageDataManifest(2, './2.jpg', 10, 10, [ImageClassificationLabelManifest(1)]),
        ]
        manifest = DatasetManifest(images, [CategoryManifest(0, 'a'), CategoryManifest(1, 'b'), CategoryManifest(2, 'c')], DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)
        dst_size = 120
        spawner = SpawnFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, SpawnConfig(0, dst_size))
        new_manifest = spawner.run(manifest)
        self.assertEqual(len(new_manifest), dst_size)

        # Generate balanced instance weights, spawn the dataset to balance classes.
        from vision_datasets.common.data_manifest import BalancedInstanceWeightsGenerator, WeightsGenerationConfig
        instance_weights = BalancedInstanceWeightsGenerator(WeightsGenerationConfig(False)).run(manifest)
        spawner = SpawnFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, SpawnConfig(0, dst_size, instance_weights))
        new_manifest = spawner.run(manifest)
        cnt = self.cnt_multiclass_labels(new_manifest)
        self.assertEqual(cnt, [40, 40, 40])

    def cnt_multiclass_labels(self, manifest):
        n_images_by_classe = [0] * len(manifest.categories) if not manifest.is_multitask else [0] * sum([len(x) for x in manifest.categories.values()])
        for im in manifest.images:
            for label in im.labels:
                n_images_by_classe[label.category_id] += 1
        return n_images_by_classe


class TestCocoGeneration(unittest.TestCase):
    def test_coco_generation(self):
        for data_type in [
                DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, DatasetTypes.IMAGE_OBJECT_DETECTION, DatasetTypes.IMAGE_CAPTION,
                DatasetTypes.IMAGE_REGRESSION, DatasetTypes.TEXT_2_IMAGE_RETRIEVAL]:
            for i in range(len(TestCases.manifest_dict_by_data_type[data_type])):
                manifest = TestCases.get_manifest(data_type, i)
                coco_dict = CocoDictGeneratorFactory.create(data_type).run(manifest)

                assert coco_dict == TestCases.manifest_dict_by_data_type[data_type][i], f'fails with {data_type} {i}'


class TestDatasetManifestMerge(unittest.TestCase):
    def test_merge_two_ic_datasets_diff_labelmap(self):
        strategy = ManifestMergeStrategyFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)
        merger = ManifestMerger(strategy)
        merged_manifest = merger.run(TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, 0),
                                     TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, 1))
        assert [c.name for c in merged_manifest.categories] == ['cat', 'dog', 'tiger', 'rabbit']
        assert [x.label_data for x in merged_manifest.images[0].labels] == [0]
        assert [x.label_data for x in merged_manifest.images[1].labels] == [0, 1]
        assert [x.label_data for x in merged_manifest.images[2].labels] == [2]
        assert [x.label_data for x in merged_manifest.images[3].labels] == [2, 3]

    def test_merge_two_ic_datasets_same_labelmap(self):
        strategy = ManifestMergeStrategyFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL)
        merger = ManifestMerger(strategy)
        merged_manifest = merger.run(TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, 0),
                                     TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, 0))
        assert [c.name for c in merged_manifest.categories] == ['cat', 'dog']
        assert [x.label_data for x in merged_manifest.images[0].labels] == [0]
        assert [x.label_data for x in merged_manifest.images[1].labels] == [0, 1]
        assert [x.label_data for x in merged_manifest.images[2].labels] == [0]
        assert [x.label_data for x in merged_manifest.images[3].labels] == [0, 1]

    def test_merge_three_ic_datasets_diff_labelmap(self):
        md3 = copy.deepcopy(TestCases.ic_multiclass_manifest_dicts[2])
        md3['categories'] = [{"id": 1, "name": "human"}, {"id": 2, "name": "snake"}]
        strategy = ManifestMergeStrategyFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS)
        merger = ManifestMerger(strategy)
        merged_manifest = merger.run(TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, 0),
                                     TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, 1),
                                     _coco_dict_to_manifest(md3, DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS))
        assert [c.name for c in merged_manifest.categories] == ['cat', 'dog', 'tiger', 'rabbit', 'human', 'snake']
        assert [x.label_data for x in merged_manifest.images[0].labels] == [0]
        assert [x.label_data for x in merged_manifest.images[1].labels] == [1]
        assert [x.label_data for x in merged_manifest.images[2].labels] == [2]
        assert [x.label_data for x in merged_manifest.images[3].labels] == [3]
        assert [x.label_data for x in merged_manifest.images[4].labels] == [4]
        assert [x.label_data for x in merged_manifest.images[5].labels] == [5]

    def test_merge_two_od_datasets_diff_labelmap(self):
        strategy = ManifestMergeStrategyFactory.create(DatasetTypes.IMAGE_OBJECT_DETECTION)
        merger = ManifestMerger(strategy)
        merged_manifest = merger.run(TestCases.get_manifest(DatasetTypes.IMAGE_OBJECT_DETECTION, 0),
                                     TestCases.get_manifest(DatasetTypes.IMAGE_OBJECT_DETECTION, 1))
        assert [c.name for c in merged_manifest.categories] == ['cat', 'dog', 'tiger', 'rabbit']
        assert [x.label_data for x in merged_manifest.images[0].labels] == [[0, 10, 10, 100, 100]]
        assert [x.label_data for x in merged_manifest.images[1].labels] == [[0, 100, 100, 200, 200], [1, 20, 20, 200, 200]]
        assert [x.label_data for x in merged_manifest.images[2].labels] == [[2, 10, 10, 90, 90]]
        assert [x.label_data for x in merged_manifest.images[3].labels] == [[2, 90, 90, 180, 180], [3, 20, 20, 200, 200]]

    def test_merge_two_od_datasets_same_labelmap(self):
        strategy = ManifestMergeStrategyFactory.create(DatasetTypes.IMAGE_OBJECT_DETECTION)
        merger = ManifestMerger(strategy)
        merged_manifest = merger.run(TestCases.get_manifest(DatasetTypes.IMAGE_OBJECT_DETECTION, 0),
                                     TestCases.get_manifest(DatasetTypes.IMAGE_OBJECT_DETECTION, 2))
        assert [c.name for c in merged_manifest.categories] == ['cat', 'dog']
        assert [x.label_data for x in merged_manifest.images[0].labels] == [[0, 10, 10, 100, 100]]
        assert [x.label_data for x in merged_manifest.images[1].labels] == [[0, 100, 100, 200, 200], [1, 20, 20, 200, 200]]
        assert [x.label_data for x in merged_manifest.images[2].labels] == [[0, 10, 10, 90, 90]]
        assert [x.label_data for x in merged_manifest.images[3].labels] == [[1, 90, 90, 180, 180]]

    def test_merge_two_caption_datasets(self):
        strategy = ManifestMergeStrategyFactory.create(DatasetTypes.IMAGE_CAPTION)
        merger = ManifestMerger(strategy)
        merged_manifest = merger.run(TestCases.get_manifest(DatasetTypes.IMAGE_CAPTION, 0),
                                     TestCases.get_manifest(DatasetTypes.IMAGE_CAPTION, 1))

        assert not merged_manifest.categories
        assert len(merged_manifest) == 4
        assert [x.label_data for x in merged_manifest.images[0].labels] == ['test 1.']
        assert [x.label_data for x in merged_manifest.images[1].labels] == ['test 2.']
        assert [x.label_data for x in merged_manifest.images[2].labels] == ['test 3.']
        assert [x.label_data for x in merged_manifest.images[3].labels] == ['test 4.']

    def test_merge_multitask_datasets_flavor0_with_same_tasks_different_types(self):
        multitask_manifest_1 = generate_multitask_dataset_manifest({
            'task1': TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, 0),
            'task2': TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, 1)
        })

        multitask_manifest_2 = generate_multitask_dataset_manifest({
            'task1': TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, 0),
            'task2': TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, 1),
        })
        strategy = ManifestMergeStrategyFactory.create(DatasetTypes.MULTITASK)
        merger = ManifestMerger(strategy)
        self.assertRaises(ValueError, lambda: merger.run(multitask_manifest_1, multitask_manifest_2))

    def test_merge_multitask_datasets_flavor0_with_different_tasks_should_raise(self):
        multitask_manifest_1 = generate_multitask_dataset_manifest({
            'task1': TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, 0),
            'task3': TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, 1)
        })

        multitask_manifest_2 = generate_multitask_dataset_manifest({
            'task1': TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, 0),
            'task2': TestCases.get_manifest(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, 1),
        })
        strategy = ManifestMergeStrategyFactory.create(DatasetTypes.MULTITASK)
        merger = ManifestMerger(strategy)
        self.assertRaises(ValueError, lambda: merger.run(multitask_manifest_1, multitask_manifest_2))


if __name__ == '__main__':
    unittest.main()
