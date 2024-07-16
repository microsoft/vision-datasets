import json
import pathlib
import tempfile
import pytest
from vision_datasets.common import DatasetTypes, CocoManifestAdaptorFactory, KVPairDatasetInfo
from .coco_adaptor_base import BaseCocoAdaptor
from ..resources.util import coco_database


class TestKVPair(BaseCocoAdaptor):
    TASK = DatasetTypes.KV_PAIR

    @pytest.mark.parametrize("coco_dict", coco_database[TASK])
    def test_create_data_manifest(self, coco_dict):
        super().test_create_data_manifest(coco_dict)
                
    @pytest.mark.parametrize("coco_dict", coco_database[TASK])
    def test_create_data_manifest_with_additional_info(self, coco_dict):
        super().test_create_data_manifest_with_additional_info(coco_dict)
        
    def test_create_data_manifest_example(self):
        coco_dict = {
            "images": [
                {
                    "id": 1,
                    "width": 224,
                    "height": 224,
                    "file_name": "train_images/1.jpg",
                    "zip_file": "train_images.zip",
                    "metadata": {
                        "catalog": True,
                        "description": "iphone 12"
                    }
                },
                {
                    "id": 2,
                    "width": 224,
                    "height": 224,
                    "file_name": "train_images/2.jpg",
                    "zip_file": "train_images.zip",
                    "metadata": {
                        "catalog": False,
                        "description": "user 1xxx's review."
                    }
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": [1, 2],
                    "key_value_pairs": {
                        "productMatch": False,
                        "rationale": "The products appear to be similar, but do not have the same brand name or text on them. The catalog image also has more than one port on the left side and a curved appearance, while the product image has ports on two sides and has a boxy appearance with no curves.",
                        "hasDamage": True,
                        "damageDetails": "Scratch on the outside"
                    }
                },
                {
                    "id": 1,
                    "image_id": [2, 1],
                    "text_input": {
                        "note": "image order is reversed"
                    },
                    "key_value_pairs": {
                        "productMatch": False,
                        "rationale": "",
                        "hasDamage": True,
                        "damageDetails": "Scratch on the outside"
                    }
                }
            ]
        }
        
        schema = {
            "name": "Retail Fraud Detection Schema",
            "description": "Schema for detecting retail fraud by comparing product images",
            "fieldSchema": {
                    "productMatch": {
                    "type": "boolean",
                    "description": "Does the product match between the two images",
                    "item": None,
                    "properties": None
                },
                "rationale": {
                    "type": "string",
                    "description": "Reason for the 'Product Match' decision",
                    "item": None,
                    "properties": None
                },
                "hasDamage": {
                    "type": "boolean",
                    "description": "Is image 2 damaged based on comparison",
                    "item": None,
                    "properties": None
                },
                "damageDetails": {
                    "type": "string",
                    "description": "Describe the damage if any in detail",
                    "item": None,
                    "properties": None
                }
            }
        }
        # schema = KVPairDatasetInfo.Schema(schema)
        adaptor = CocoManifestAdaptorFactory.create(TestKVPair.TASK, schema=schema)
        with tempfile.TemporaryDirectory() as temp_dir:
            dm1_path = pathlib.Path(temp_dir) / 'coco.json'
            dm1_path.write_text(json.dumps(coco_dict))
            kv_pair_manifest = adaptor.create_dataset_manifest(str(dm1_path))
            
        kv_pair_manifest.images[0].additional_info['meta_data'] = coco_dict['images'][0]['metadata']
        
        ann_0 = kv_pair_manifest.annotations[0]
        assert ann_0.id == coco_dict['annotations'][0]['id']
        assert ann_0.img_ids == [0, 1]
        assert ann_0.label.key_value_pairs == coco_dict['annotations'][0]['key_value_pairs']
        assert ann_0.label.text_input is None
        
        ann_1 = kv_pair_manifest.annotations[1]
        assert ann_1.id == coco_dict['annotations'][1]['id']
        assert ann_1.img_ids == [1, 0]
        assert ann_1.label.key_value_pairs == coco_dict['annotations'][1]['key_value_pairs']
        assert ann_1.label.text_input == {"note": "image order is reversed"}
        