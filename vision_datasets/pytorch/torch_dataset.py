import logging

from .dataset import Dataset

logger = logging.getLogger(__name__)


class TorchDataset(Dataset):
    """
    Dataset class used for pytorch training
    """

    def __init__(self, manifest_dataset, transform=None):
        Dataset.__init__(self, transform)
        self.dataset = manifest_dataset
        self.dataset_info = manifest_dataset.dataset_info

    @property
    def labels(self):
        return self.dataset.labels

    @property
    def dataset_resources(self):
        return self.dataset.dataset_resources

    def __getitem__(self, index):
        if isinstance(index, int):
            image, target, idx_str = self.dataset[index]
            image, target = self.transform(image, target)
            return image, target, idx_str
        else:
            return [self.transform(img, target) + (idx,) for img, target, idx in self.dataset[index]]

    def __len__(self):
        return len(self.dataset)

    def close(self):
        self.dataset.close()
