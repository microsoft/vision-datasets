import copy
import json
import pathlib
import tempfile

from PIL import Image

from vision_datasets import DatasetInfo, CocoManifestAdaptor, DatasetTypes, ManifestDataset


class DetectionTestFixtures:
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
    def create_an_od_manifest(root_dir=''):
        images = [{'id': i + 1, 'file_name': f'{i + 1}.jpg', 'width': 100, 'height': 100} for i in range(2)]

        categories = [{'id': i + 1, 'name': f'{i + 1}-class', } for i in range(4)]

        annotations = [
            {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 100, 100]},
            {'id': 2, 'image_id': 1, 'category_id': 2, 'bbox': [10, 10, 40, 90]},
            {'id': 3, 'image_id': 2, 'category_id': 3, 'bbox': [50, 50, 30, 30]},
            {'id': 4, 'image_id': 2, 'category_id': 4, 'bbox': [0, 50, 100, 50]}
        ]

        coco_dict = {'images': images, 'categories': categories, 'annotations': annotations}
        coco_path = pathlib.Path(root_dir) / 'coco.json'
        coco_path.write_text(json.dumps(coco_dict))
        return CocoManifestAdaptor.create_dataset_manifest(coco_path.name, DatasetTypes.IC_MULTICLASS, root_dir)

    @staticmethod
    def create_an_od_dataset(n_images=2, coordinates='relative'):
        dataset_dict = copy.deepcopy(DetectionTestFixtures.DATASET_INFO_DICT)

        tempdir = tempfile.TemporaryDirectory()
        dataset_dict['root_folder'] = tempdir.name
        dataset_dict['type'] = 'object_detection'
        for i in range(n_images):
            Image.new('RGB', (100, 100)).save(pathlib.Path(tempdir.name) / f'{i + 1}.jpg')
            print(pathlib.Path(tempdir.name) / f'{i + 1}.jpg')

        dataset_info = DatasetInfo(dataset_dict)
        dataset_manifest = DetectionTestFixtures.create_an_od_manifest(tempdir.name)
        dataset = ManifestDataset(dataset_info, dataset_manifest, coordinates)
        return dataset, tempdir
