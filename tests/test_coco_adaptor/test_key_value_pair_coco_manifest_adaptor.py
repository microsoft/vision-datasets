import copy
import json
import tempfile
import pathlib
import pytest

from vision_datasets.common import DatasetTypes, CocoManifestAdaptorFactory
from .coco_adaptor_base import BaseCocoAdaptor
from ..resources.util import coco_database, schema_database


class TestKeyValuePair(BaseCocoAdaptor):
    TASK = DatasetTypes.KEY_VALUE_PAIR

    @pytest.mark.parametrize("coco_dict, schema", zip(coco_database[TASK], schema_database[TASK]))
    def test_create_data_manifest(self, coco_dict, schema):
        super().test_create_data_manifest(coco_dict, schema)

    @pytest.mark.parametrize("coco_dict, schema", zip(coco_database[TASK], schema_database[TASK]))
    def test_create_data_manifest_with_additional_info(self, coco_dict, schema):
        super().test_create_data_manifest_with_additional_info(coco_dict, schema)

    def prepare_schema_and_coco_dict(self):
        schema = copy.deepcopy(schema_database[TestKeyValuePair.TASK][1])
        coco_dict = copy.deepcopy(coco_database[TestKeyValuePair.TASK][1])
        return schema, coco_dict

    def test_create_data_manifest_example(self):
        schema, coco_dict = self.prepare_schema_and_coco_dict()
        adaptor = CocoManifestAdaptorFactory.create(TestKeyValuePair.TASK, schema=schema)
        with tempfile.TemporaryDirectory() as temp_dir:
            dm1_path = pathlib.Path(temp_dir) / 'coco.json'
            dm1_path.write_text(json.dumps(coco_dict))
            key_value_pair_manifest = adaptor.create_dataset_manifest(str(dm1_path))

        key_value_pair_manifest.images[0].additional_info['meta_data'] = coco_dict['images'][0]['metadata']

        ann_0 = key_value_pair_manifest.annotations[0]
        assert ann_0.id == coco_dict['annotations'][0]['id']
        assert ann_0.img_ids == [0, 1]
        assert ann_0.fields == coco_dict['annotations'][0]['fields']
        assert ann_0.text is None

        ann_1 = key_value_pair_manifest.annotations[1]
        assert ann_1.id == coco_dict['annotations'][1]['id']
        assert ann_1.img_ids == [1, 0]
        assert ann_1.fields == coco_dict['annotations'][1]['fields']
        assert ann_1.text == coco_dict['annotations'][1]['text']
        
    def test_schema_mismatch_key_value_pair(self):
        schema, coco_dict = self.prepare_schema_and_coco_dict()
        # remove a field that defined in schema
        del coco_dict['annotations'][0]['fields']['productMatch']

        adaptor = CocoManifestAdaptorFactory.create(TestKeyValuePair.TASK, schema=schema)
        with tempfile.TemporaryDirectory() as temp_dir:
            dm1_path = pathlib.Path(temp_dir) / 'coco.json'
            dm1_path.write_text(json.dumps(coco_dict))
            with pytest.raises(ValueError):
                adaptor.create_dataset_manifest(str(dm1_path))

    def test_grounding_error_for_multi_image_annotation(self):
        schema = copy.deepcopy(schema_database[TestKeyValuePair.TASK][2])
        coco_dict = copy.deepcopy(coco_database[TestKeyValuePair.TASK][2])
        # change one annotation to multi-image annotation
        coco_dict['annotations'][0]['image_ids'] = [1, 2]
        
        adaptor = CocoManifestAdaptorFactory.create(TestKeyValuePair.TASK, schema=schema)
        with tempfile.TemporaryDirectory() as temp_dir:
            dm1_path = pathlib.Path(temp_dir) / 'coco.json'
            dm1_path.write_text(json.dumps(coco_dict))
            with pytest.raises(ValueError):
                adaptor.create_dataset_manifest(str(dm1_path))
