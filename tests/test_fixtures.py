import copy
import json
import pathlib
import tempfile

from PIL import Image

from vision_datasets.common import CocoManifestAdaptorFactory, DatasetInfo, DatasetTypes, VisionDataset


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
