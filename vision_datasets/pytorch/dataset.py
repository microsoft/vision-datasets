from PIL import ImageFile
from abc import ABC
import torch
from inspect import signature


class Dataset(torch.utils.data.Dataset, ABC):
    def __init__(self, transform=None):
        super().__init__()

        if not transform:
            self._transform = lambda x, y: (x, y)
        elif len(signature(transform).parameters) == 1:
            self._transform = lambda x, y: (transform(x), y)
        else:
            self._transform = transform

        # Work around for corrupted files in datasets
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    @property
    def labels(self):
        """Returns a list of labels."""
        raise NotImplementedError

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, val):
        self._transform = val if val else lambda x, y: (x, y)

    def close(self):
        """Release the resources allocated for this dataset."""
        pass

    def get_original_image_count(self):
        return len(self)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
