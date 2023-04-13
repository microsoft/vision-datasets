import pytest
from vision_datasets.common.manifest_v15.coco_data_manifest_adaptor import ManifestAdaptorFactory
from vision_datasets import DatasetTypes
from .util import coco_database, coco_dict_to_manifest


def rough_check(manifest, coco_dict):
    assert len(manifest.images) == len(coco_dict['images'])
    categories = coco_dict.get('categories')
    if categories:
        assert manifest.labelmap and len(manifest.labelmap) == len(categories)
    assert sum([len(img.labels) for img in manifest.images]) == len(coco_dict['annotations'])


@pytest.mark.parametrize("task, coco_dict", [(task, coco_dict) for task, coco_dicts in coco_database.items() for coco_dict in coco_dicts])
def test_create_data_manifest(task, coco_dict):
    adaptor = ManifestAdaptorFactory.create(task)
    manifest = coco_dict_to_manifest(adaptor, coco_dict, task)
    rough_check(manifest, coco_dict)
