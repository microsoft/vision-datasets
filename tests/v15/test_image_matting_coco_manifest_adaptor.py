import pytest
import pathlib
from vision_datasets.common.manifest_v15.coco_data_manifest_adaptor import ManifestAdaptorFactory, DatasetTypes
from .util import coco_dict_to_manifest


class ImageMattingTestCases:
    root_path = str(pathlib.Path(__file__).resolve().parent.parent)

    manifest_dicts = [
        {
            "images": [{"id": 1, "file_name": "train_images.zip@image/test_1.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "label": f"{root_path}/image_matting_test_data.zip@mask/test_1.png"}
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@image/test_1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@image/test_2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "label": f"{root_path}/image_matting_test_data.zip@mask/test_1.png"},
                {"id": 2, "image_id": 2, "label": f"{root_path}/image_matting_test_data.zip@mask/test_2.png"},
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@image/test_1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@image/test_2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "label": f"{root_path}/image_matting_test_data.zip@mask/test_1.png"},
                {"id": 2, "image_id": 2, "label": f"{root_path}/image_matting_test_data.zip@mask/test_2.png"},
            ]
        }]


@pytest.mark.parametrize("coco_dict", ImageMattingTestCases.manifest_dicts)
def test_create_data_manifest(coco_dict):
    adaptor = ManifestAdaptorFactory.create(DatasetTypes.IMAGE_MATTING)
    coco_dict_to_manifest(adaptor, coco_dict, DatasetTypes.IMAGE_MATTING)
