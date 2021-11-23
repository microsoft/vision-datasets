import pickle
import unittest
from vision_datasets.pytorch import TorchDataset


class FakeDatasetInfo:
    @property
    def dataset_info(self):
        return None


def _one_arg_method(x):
    return x


class TestPytorchDataset(unittest.TestCase):
    def test_picklable(self):
        dataset = TorchDataset(FakeDatasetInfo())
        serialized = pickle.dumps(dataset)
        new_dataset = pickle.loads(serialized)
        self.assertIsInstance(new_dataset, TorchDataset)

        dataset = TorchDataset(FakeDatasetInfo(), _one_arg_method)
        serialized = pickle.dumps(dataset)
        new_dataset = pickle.loads(serialized)
        self.assertIsInstance(new_dataset, TorchDataset)

        dataset = TorchDataset(FakeDatasetInfo())
        dataset.transform = None
        serialized = pickle.dumps(dataset)
        new_dataset = pickle.loads(serialized)
        self.assertIsInstance(new_dataset, TorchDataset)


if __name__ == '__main__':
    unittest.main()
