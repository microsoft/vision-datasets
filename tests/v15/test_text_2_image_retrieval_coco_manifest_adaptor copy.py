import pytest
from vision_datasets.common.manifest_v15.coco_data_manifest_adaptor import ManifestAdaptorFactory, DatasetTypes
from .util import coco_dict_to_manifest


class Text2ImageRetrievalTestCases:
    manifest_dicts = [
        {
            "images": [
                {"id": 1, "file_name": "test1.zip@test/0/image_1.jpg"}, {"id": 2, "file_name": "test2.zip@test/1/image_2.jpg"}
            ],
            "annotations": [
                {"image_id": 1, "id": 1, "category_id": 1, "query": "apple"},
                {"image_id": 2, "id": 2, "category_id": 2, "query": "banana"}
            ]
        },
        {
            "images": [
                {"id": 1, "file_name": "test1.zip@test/0/image_1.jpg"}, {"id": 2, "file_name": "test2.zip@test/1/image_2.jpg"}
            ],
            "annotations": [
                {"image_id": 1, "id": 1, "query": "apple"},
                {"image_id": 2, "id": 2, "query": "banana"}
            ]
        }
    ]


@pytest.mark.parametrize("coco_dict", Text2ImageRetrievalTestCases.manifest_dicts)
def test_create_data_manifest(coco_dict):
    adaptor = ManifestAdaptorFactory.create(DatasetTypes.IMAGE_RETRIEVAL)
    coco_dict_to_manifest(adaptor, coco_dict, DatasetTypes.IMAGE_RETRIEVAL)
