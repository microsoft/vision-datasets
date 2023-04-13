import pytest
from vision_datasets.common.manifest_v15.coco_data_manifest_adaptor import ManifestAdaptorFactory, DatasetTypes
from .util import coco_dict_to_manifest


class MultiClassClassificationTestCases:
    manifest_dicts = [
        {
            "images": [
                {"id": 1, "width": 224.0, "height": 224.0, "file_name": "train/1.jpg"},
                {"id": 2, "width": 224.0, "height": 224.0, "file_name": "train/3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1},
                {"id": 2, "category_id": 2, "image_id": 1},
                {"id": 3, "category_id": 2, "image_id": 2}
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        }
    ]


@pytest.mark.parametrize("coco_dict", MultiClassClassificationTestCases.manifest_dicts)
def test_create_data_manifest_when_multiple_annotation_per_image_should_fail(coco_dict):
    adaptor = ManifestAdaptorFactory.create(DatasetTypes.IC_MULTICLASS)
    with pytest.raises(AssertionError, match='There should be exactly one label per image for classification_multiclass datasets, but image with id 1 has more than one.'):
        coco_dict_to_manifest(adaptor, coco_dict, DatasetTypes.IC_MULTICLASS)
