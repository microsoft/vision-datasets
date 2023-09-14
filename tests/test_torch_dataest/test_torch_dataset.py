import json
import os
import pathlib
import pickle
import tempfile

import pytest
from PIL import Image

from vision_datasets import DatasetInfo, DatasetTypes, VisionDataset
from vision_datasets.torch import TorchDataset

from ..resources.util import coco_database, coco_dict_to_manifest


class FakeDataset:
    pass


def _one_arg_method(x):
    return x


class TestTorchDataset:
    @pytest.mark.parametrize("data_type, coco_dicts", [(data_type, coco_database[data_type]) for data_type in DatasetTypes
                                                       if data_type != DatasetTypes.MULTITASK])
    def test_create_torch_dataset(self, data_type, coco_dicts):
        coco_dict = coco_dicts[0]
        manifest = coco_dict_to_manifest(data_type, coco_dict)
        with tempfile.TemporaryDirectory() as temp_dir:
            tdir = pathlib.Path(temp_dir)
            (tdir / 'test.json').write_text(json.dumps(coco_dict))

            dataset_info = DatasetInfo({
                'name': 'test',
                'type': data_type.name,
                'root_folder': tdir.as_posix(),
                'format': 'coco',
                'train': {'index_path': 'test.json'}
            })
            td = TorchDataset(VisionDataset(dataset_info, manifest))

            for image in manifest.images:
                image.img_path = image.img_path.split('@')[1] if '@' in image.img_path else image.img_path
                image.img_path = tdir / image.img_path
                os.makedirs(image.img_path.parent, exist_ok=True)
                image.img_path = image.img_path.as_posix()
                Image.new(mode="RGB", size=(20, 20)).save(image.img_path)

            for x in td:
                pass

            td[0:-1]

    def test_picklable(self):
        dataset = TorchDataset(FakeDataset())
        serialized = pickle.dumps(dataset)
        new_dataset = pickle.loads(serialized)
        assert isinstance(new_dataset, TorchDataset)

        dataset = TorchDataset(FakeDataset(), _one_arg_method)
        serialized = pickle.dumps(dataset)
        new_dataset = pickle.loads(serialized)
        assert isinstance(new_dataset, TorchDataset)

        dataset = TorchDataset(FakeDataset())
        dataset.transform = None
        serialized = pickle.dumps(dataset)
        new_dataset = pickle.loads(serialized)
        assert isinstance(new_dataset, TorchDataset)

    def test_transform(self):
        dataset = TorchDataset(FakeDataset(), None)
        assert dataset.transform(1, 2) == (1, 2)
        dataset = TorchDataset(FakeDataset(), lambda x: x)
        assert dataset.transform(1, 2) == (1, 2)
        dataset = TorchDataset(FakeDataset(), lambda x, y: (x, y))
        assert dataset.transform(1, 2) == (1, 2)

        dataset.transform = lambda x: x
        assert dataset.transform(1, 2) == (1, 2)
