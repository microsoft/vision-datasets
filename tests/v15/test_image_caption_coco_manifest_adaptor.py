import pytest
from vision_datasets.common.manifest_v15.coco_data_manifest_adaptor import ManifestAdaptorFactory, DatasetTypes
from .util import coco_dict_to_manifest


class ImageCaptionTestCases:
    manifest_dicts = [
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


@pytest.mark.parametrize("coco_dict", ImageCaptionTestCases.manifest_dicts)
def test_create_data_manifest(coco_dict):
    adaptor = ManifestAdaptorFactory.create(DatasetTypes.IMCAP)
    coco_dict_to_manifest(adaptor, coco_dict, DatasetTypes.IMCAP)
