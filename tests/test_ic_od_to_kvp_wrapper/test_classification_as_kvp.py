import copy
import json
import pathlib
import tempfile
import unittest

from PIL import Image

from vision_datasets.common import (
    CocoManifestAdaptorFactory,
    DatasetInfo,
    DatasetTypes,
    VisionDataset,
)
from vision_datasets.image_classification.classification_as_kvp_dataset import (
    ClassificationAsKeyValuePairDataset,
)


class TestMultilcassClassificationDataset:
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
        dataset_dict = copy.deepcopy(TestMultilcassClassificationDataset.DATASET_INFO_DICT)
        tempdir = tempfile.TemporaryDirectory()
        dataset_dict['root_folder'] = tempdir.name
        for i in range(n_images):
            Image.new('RGB', (100, 100)).save(pathlib.Path(tempdir.name) / f'{i + 1}.jpg')

        dataset_info = DatasetInfo(dataset_dict)
        dataset_manifest = TestMultilcassClassificationDataset().create_an_ic_manifest(tempdir.name, n_images, n_categories)
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


class TestClassificationAsKeyValuePairDataset(unittest.TestCase):
    def test_simple(self):
        sample_classification_dataset, _ = TestMultilcassClassificationDataset.create_an_ic_dataset()
        kvp_dataset = ClassificationAsKeyValuePairDataset(sample_classification_dataset)

        print(kvp_dataset)

        self.assertIsInstance(kvp_dataset, ClassificationAsKeyValuePairDataset)
        self.assertEqual(kvp_dataset.dataset_info.type, DatasetTypes.KEY_VALUE_PAIR)
        self.assertIn("name", kvp_dataset.dataset_info.schema)
        self.assertIn("description", kvp_dataset.dataset_info.schema)
        self.assertIn("fieldSchema", kvp_dataset.dataset_info.schema)

        self.assertEqual(kvp_dataset.dataset_info.schema["fieldSchema"],
                         {"className": {
                             "type": "string",
                             "description": "Class name that the image belongs to.",
                             "classes": {
                                 "1-class": {},
                                 "2-class": {},
                                 "3-class": {},
                             }
                         }
        })


if __name__ == '__main__':
    unittest.main()
