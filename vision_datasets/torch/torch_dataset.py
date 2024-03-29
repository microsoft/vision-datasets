import logging

from .dataset import Dataset
from ..common.dataset.vision_dataset import VisionDataset

logger = logging.getLogger(__name__)


class TorchDataset(Dataset):
    """
    Dataset class used for pytorch training
    """

    def __init__(self, manifest_dataset: VisionDataset, transform=None):
        Dataset.__init__(self, transform)
        self.dataset = manifest_dataset

    @property
    def categories(self):
        return self.dataset.categories

    @property
    def dataset_resources(self):
        return self.dataset.dataset_resources

    @property
    def dataset_info(self):
        return self.dataset.dataset_info

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
