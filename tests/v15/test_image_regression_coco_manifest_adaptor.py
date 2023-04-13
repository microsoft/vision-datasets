import pytest
from vision_datasets.common.manifest_v15.coco_data_manifest_adaptor import ManifestAdaptorFactory, DatasetTypes
from .util import coco_dict_to_manifest


class ImageRegressionTestCases:
    manifest_dicts = [
        {
            "images": [{"id": 1, "file_name": "train_images.zip@1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "target": 1.0},
                {"id": 2, "image_id": 1, "target": 3.0},
                {"id": 3, "image_id": 2, "target": 2.0},
            ]
        }]


@pytest.mark.parametrize("coco_dict", ImageRegressionTestCases.manifest_dicts)
def test_create_data_manifest_when_multiple_annotation_per_image_should_fail(coco_dict):
    adaptor = ManifestAdaptorFactory.create(DatasetTypes.IMAGE_REGRESSION)
    with pytest.raises(AssertionError, match='There should be exactly one label per image for image_regression datasets, but image with id 1 has more than one.'):
        coco_dict_to_manifest(adaptor, coco_dict, DatasetTypes.IMAGE_REGRESSION)
