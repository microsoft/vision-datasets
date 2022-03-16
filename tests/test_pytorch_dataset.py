import pickle
import unittest

from vision_datasets.pytorch import TorchDataset


class FakeDataset:
    pass


def _one_arg_method(x):
    return x


class TestPytorchDataset(unittest.TestCase):
    def test_picklable(self):
        dataset = TorchDataset(FakeDataset())
        serialized = pickle.dumps(dataset)
        new_dataset = pickle.loads(serialized)
        self.assertIsInstance(new_dataset, TorchDataset)

        dataset = TorchDataset(FakeDataset(), _one_arg_method)
        serialized = pickle.dumps(dataset)
        new_dataset = pickle.loads(serialized)
        self.assertIsInstance(new_dataset, TorchDataset)

        dataset = TorchDataset(FakeDataset())
        dataset.transform = None
        serialized = pickle.dumps(dataset)
        new_dataset = pickle.loads(serialized)
        self.assertIsInstance(new_dataset, TorchDataset)

    def test_transform(self):
        dataset = TorchDataset(FakeDataset(), None)
        assert dataset.transform(1, 2) == (1, 2)
        dataset = TorchDataset(FakeDataset(), lambda x: x)
        assert dataset.transform(1, 2) == (1, 2)
        dataset = TorchDataset(FakeDataset(), lambda x, y: (x, y))
        assert dataset.transform(1, 2) == (1, 2)

        dataset.transform = lambda x: x
        assert dataset.transform(1, 2) == (1, 2)


if __name__ == '__main__':
    unittest.main()
