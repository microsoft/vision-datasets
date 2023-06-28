import json
import unittest

from vision_datasets.common import DatasetRegistry, DatasetTypes, Usages


class TestDatasetRegistry(unittest.TestCase):
    DUMMY_DATA_1 = {
        "name": "dummy1",
        "version": 1,
        "type": "classification_multiclass",
        "root_folder": "dummy",
        "test": {
            "index_path": "test.txt",
            "files_for_local_usage": [
                "Train.zip"
            ]
        }
    }

    DUMMY_DATA_1_V2 = {
        "name": "dummy2",
        "version": 1,
        "type": "classification_multiclass",
        "root_folder": "dummy",
        "test": {
            "index_path": "test.txt",
            "files_for_local_usage": [
                "Train.zip"
            ]
        }
    }

    DUMMY_DATA_2 = {
        "name": "dummy2",
        "version": 1,
        "type": "classification_multiclass",
        "root_folder": "dummy",
        "test": {
            "index_path": "test.txt",
            "files_for_local_usage": [
                "Train.zip"
            ]
        }
    }

    def test_create_dataset_reg(self):
        dn = self.DUMMY_DATA_1['name']
        dr = DatasetRegistry(json.dumps([self.DUMMY_DATA_1]))
        assert len(dr.list_data_version_and_types()) == 1
        info = dr.get_dataset_info(dn)
        assert info
        assert info.name == dn
        assert info.version == self.DUMMY_DATA_1['version']
        assert info.root_folder == self.DUMMY_DATA_1['root_folder']
        assert info.type == DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS
        assert info.files_for_local_usage[Usages.TEST] == self.DUMMY_DATA_1['test']['files_for_local_usage']
        assert info.index_files[Usages.TEST] == self.DUMMY_DATA_1['test']['index_path']

    def test_create_dataset_reg_with_two_jsons(self):
        dr = DatasetRegistry([json.dumps([self.DUMMY_DATA_1]), json.dumps([self.DUMMY_DATA_2])])
        assert len(dr.list_data_version_and_types()) == 2
        assert dr.get_dataset_info(self.DUMMY_DATA_1['name'])
        assert dr.get_dataset_info(self.DUMMY_DATA_2['name'])
