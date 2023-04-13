import pytest
from vision_datasets.common.manifest_v15.coco_data_manifest_adaptor import ManifestAdaptorFactory, DatasetTypes
from .util import coco_dict_to_manifest


class MultiLabelClassificationTestCases:
    manifest_dicts = [
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


@pytest.mark.parametrize("coco_dict", MultiLabelClassificationTestCases.manifest_dicts)
def test_create_data_manifest(coco_dict):
    adaptor = ManifestAdaptorFactory.create(DatasetTypes.IC_MULTILABEL)
    manifest = coco_dict_to_manifest(adaptor, coco_dict, DatasetTypes.IC_MULTILABEL)
    assert len(manifest.images) == len(coco_dict['images'])
    assert len(manifest.labelmap) == len(coco_dict['categories'])
    assert sum([len(img.labels) for img in manifest.images]) == len(coco_dict['annotations'])
