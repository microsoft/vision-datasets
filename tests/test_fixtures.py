import copy
import json
import pathlib
import tempfile

from PIL import Image

from vision_datasets.common import (
    CocoManifestAdaptorFactory,
    DatasetInfo,
    DatasetTypes,
    VisionDataset,
)


class DetectionTestFixtures:
    DATASET_INFO_DICT = {
        "name": "dummy",
        "version": 1,
        "type": "object_detection",
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
    def create_an_od_manifest(root_dir='', n_images=2, n_categories=4):
        images = [{'id': i + 1, 'file_name': f'{i + 1}.jpg', 'width': 100, 'height': 100} for i in range(n_images)]

        categories = [{'id': i + 1, 'name': f'{i + 1}-class', } for i in range(n_categories)]

        bbox_set = [[0, 0, 100, 100], [10, 10, 40, 90], [50, 50, 30, 30], [0, 50, 100, 50]]
        annotations = [{'id': i + 1, 'image_id': i // 2 + 1, 'category_id': i % n_categories + 1, 'bbox': bbox_set[i % len(bbox_set)]} for i in range(n_images * 2)]

        coco_dict = {'images': images, 'categories': categories, 'annotations': annotations}
        coco_path = pathlib.Path(root_dir) / 'coco.json'
        coco_path.write_text(json.dumps(coco_dict))
        return CocoManifestAdaptorFactory.create(DatasetTypes.IMAGE_OBJECT_DETECTION).create_dataset_manifest(coco_path.name, root_dir)

    @staticmethod
    def create_an_od_dataset(n_images=2, n_categories=4, coordinates='relative'):
        dataset_dict = copy.deepcopy(DetectionTestFixtures.DATASET_INFO_DICT)
        tempdir = tempfile.TemporaryDirectory()
        dataset_dict['root_folder'] = tempdir.name
        dataset_dict['type'] = 'object_detection'
        for i in range(n_images):
            Image.new('RGB', (100, 100)).save(pathlib.Path(tempdir.name) / f'{i + 1}.jpg')

        dataset_info = DatasetInfo(dataset_dict)
        dataset_manifest = DetectionTestFixtures.create_an_od_manifest(tempdir.name, n_images, n_categories)
        dataset = VisionDataset(dataset_info, dataset_manifest, coordinates)
        return dataset, tempdir


class MulticlassClassificationTestFixtures:
    DATASET_INFO_DICT = {
        "name": "dummy",
        "version": 1,
        "type": "image_classification_multiclass",
        "root_folder": "dummy",
        "format": "coco",
        "test": {
            "index_path": "train.json",
            "files_for_local_usage": [
                "train.zip"
            ]
        },
    }

    @staticmethod
    def create_an_ic_dataset(n_images=2, n_categories=3):
        dataset_dict = copy.deepcopy(MulticlassClassificationTestFixtures.DATASET_INFO_DICT)
        tempdir = tempfile.TemporaryDirectory()
        dataset_dict['root_folder'] = tempdir.name
        for i in range(n_images):
            Image.new('RGB', (100, 100)).save(pathlib.Path(tempdir.name) / f'{i + 1}.jpg')

        dataset_info = DatasetInfo(dataset_dict)
        dataset_manifest = MulticlassClassificationTestFixtures.create_an_ic_manifest(tempdir.name, n_images, n_categories)
        dataset = VisionDataset(dataset_info, dataset_manifest)
        return dataset, tempdir

    @staticmethod
    def create_an_ic_manifest(root_dir='', n_images=2, n_categories=3):
        images = [{'id': i + 1, 'file_name': f'{i + 1}.jpg', 'width': 100, 'height': 100} for i in range(n_images)]

        categories = [{'id': i + 1, 'name': f'{i + 1}-class', } for i in range(n_categories)]

        annotations = [{'id': i + 1, 'image_id': i + 1, 'category_id': i + 1} for i in range(n_images)]

        coco_dict = {'images': images, 'categories': categories, 'annotations': annotations}
        coco_path = pathlib.Path(root_dir) / 'coco.json'
        coco_path.write_text(json.dumps(coco_dict))
        return CocoManifestAdaptorFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS).create_dataset_manifest(coco_path.name, root_dir)


class MultilabelClassificationTestFixtures:
    DATASET_INFO_DICT = {
        "name": "dummy",
        "version": 1,
        "type": "image_classification_multilabel",
        "root_folder": "dummy",
        "format": "coco",
        "test": {
            "index_path": "train.json",
            "files_for_local_usage": [
                "train.zip"
            ]
        },
    }

    @staticmethod
    def create_an_ic_dataset(n_images=2, n_categories=3):
        tempdir = tempfile.TemporaryDirectory()
        dataset_dict = copy.deepcopy(MultilabelClassificationTestFixtures.DATASET_INFO_DICT)
        dataset_dict['root_folder'] = tempdir.name
        for i in range(n_images):
            Image.new('RGB', (100, 100)).save(pathlib.Path(tempdir.name) / f'{i + 1}.jpg')

        dataset_info = DatasetInfo(dataset_dict)
        dataset_manifest = MultilabelClassificationTestFixtures.create_an_ic_manifest(tempdir.name, n_images, n_categories)
        dataset = VisionDataset(dataset_info, dataset_manifest)
        return dataset, tempdir

    @staticmethod
    def create_an_ic_manifest(root_dir='', n_images=2, n_categories=3):
        images = [{'id': i + 1, 'file_name': f'{i + 1}.jpg', 'width': 100, 'height': 100} for i in range(n_images)]

        categories = [{'id': i + 1, 'name': f'{i + 1}-class', } for i in range(n_categories)]
        annotations = [{'id': i + 1, 'image_id': i + 1, 'category_id': i + 1} for i in range(n_images)]
        annotations.extend([{'id': n_images + i + 1, 'image_id': i + 1, 'category_id': n_images - i} for i in range(n_images)])
        coco_dict = {'images': images, 'categories': categories, 'annotations': annotations}
        coco_path = pathlib.Path(root_dir) / 'coco.json'
        coco_path.write_text(json.dumps(coco_dict))
        return CocoManifestAdaptorFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL).create_dataset_manifest(coco_path.name, root_dir)


class VQATestFixtures:
    DATASET_INFO_DICT = {
        "name": "dummy",
        "version": 1,
        "type": "visual_question_answering",
        "root_folder": "dummy",
        "format": "coco",
        "test": {
            "index_path": "train.json",
            "files_for_local_usage": [
                "train.zip"
            ]
        },
    }

    @staticmethod
    def create_a_vqa_dataset(n_images=2):
        dataset_dict = copy.deepcopy(VQATestFixtures.DATASET_INFO_DICT)
        tempdir = tempfile.TemporaryDirectory()
        dataset_dict['root_folder'] = tempdir.name
        for i in range(n_images):
            Image.new('RGB', (min(1000, (i+1) * 100), min(1000, (i+1) * 100))).save(pathlib.Path(tempdir.name) / f'{i + 1}.jpg')

        dataset_info = DatasetInfo(dataset_dict)
        dataset_manifest = VQATestFixtures().create_a_vqa_manifest(tempdir.name, n_images)
        dataset = VisionDataset(dataset_info, dataset_manifest)
        return dataset, tempdir

    @staticmethod
    def create_a_vqa_manifest(root_dir='', n_images=2):
        images = [{'id': i + 1, 'file_name': f'{i + 1}.jpg', 'width': min(1000, (i+1)*100), 'height': min(1000, (i+1)*100)} for i in range(n_images)]
        annotations = [{'id': i + 1, 'image_id': i + 1, 'question': f'question {i+1}', 'answer': f'answer {i+1}'} for i in range(n_images)]

        # Add a second question for the last image
        annotations.append({'id': n_images + 1, 'image_id': n_images, 'question': 'question 3', 'answer': 'answer 3'})

        coco_dict = {'images': images, 'annotations': annotations}
        coco_path = pathlib.Path(root_dir) / 'coco.json'
        coco_path.write_text(json.dumps(coco_dict))
        return CocoManifestAdaptorFactory.create(DatasetTypes.VISUAL_QUESTION_ANSWERING).create_dataset_manifest(coco_path.name, root_dir)
