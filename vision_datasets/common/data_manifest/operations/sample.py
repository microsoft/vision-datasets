import abc
import collections
import copy
import logging
import random
import typing
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..data_manifest import DatasetManifest
from .operation import Operation

logger = logging.getLogger(__name__)


@dataclass
class SampleBaseConfig:
    random_seed: int


@dataclass
class SampleByFewShotConfig(SampleBaseConfig):
    n_shots: typing.List[int]


@dataclass
class SampleByNumSamplesConfig(SampleBaseConfig):
    with_replacement: bool
    n_samples: int
    weights: typing.List[float] = None


class SampleStrategy(abc.ABC):
    def __init__(self, config: SampleBaseConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def sample(self, manifest: DatasetManifest):
        pass


class SampleStrategyType(Enum):
    FewShot = 0  # few shot based on categories
    NumSamples = 1  # sample by target numebr of samples


class ManifestSampler(Operation):
    def __init__(self, strategy: SampleStrategy) -> None:
        super().__init__()
        self.strategy = strategy

    def run(self, *args: DatasetManifest):
        if len(args) != 1:
            raise ValueError

        return self.strategy.sample(args[0])


class SampleByNumSamples(SampleStrategy):
    """
    Downsample a dataset to a desired number of images
    """

    def __init__(self, config: SampleByNumSamplesConfig) -> None:
        if config.n_samples <= 0:
            raise ValueError('n samples must be greater than zero.')

        super().__init__(config)

    def sample(self, manifest: DatasetManifest):
        if not self.config.with_replacement and self.config.n_samples > len(manifest.images):
            raise ValueError('n_samples must be less than or equal to the number of images in the dataset.')

        rng = np.random.default_rng(self.config.random_seed)
        normalized_weights = [w / sum(self.config.weights) for w in self.config.weights] if self.config.weights else None
        sampled_indices = rng.choice(len(manifest.images), size=self.config.n_samples, replace=self.config.with_replacement, p=normalized_weights)
        sampled_images = [manifest.images[i] for i in sampled_indices]

        return DatasetManifest(copy.deepcopy(sampled_images), copy.deepcopy(manifest.categories), copy.deepcopy(manifest.data_type), copy.deepcopy(manifest.additional_info))


class SampleFewShot(SampleStrategy):
    """Greedy few-shots sampling method.
        Randomly pick images from the original datasets until all classes have at least {num_min_images_per_class} tags/boxes.

        Note that images without any tag/box will be ignored. All images in the subset will have at least one tag/box.
    """

    def __init__(self, config: SampleByFewShotConfig) -> None:
        if config.n_shots <= 0:
            raise ValueError('n shots must be greater than zero.')
        super().__init__(config)

    def sample(self, manifest: DatasetManifest):
        """
        Args:
            manifest (DatasetManifest): manifest to be sampled from.

        Returns:
            A samped dataset (DatasetManifest)

        Raises:
            RuntimeError if it couldn't find n_shots samples for all classes
        """

        images = list(manifest.images)
        rng = random.Random(self.config.random_seed)
        rng.shuffle(images)

        num_classes = len(manifest.categories) if not manifest.is_multitask else sum(len(x) for x in manifest.categories.values())
        total_counter = collections.Counter({i: self.config.n_shots for i in range(num_classes)})
        sampled_images = []
        for image in images:
            counts = collections.Counter([c.category_id for c in image.labels])
            if set((+total_counter).keys()) & set(counts.keys()):
                total_counter -= counts
                sampled_images.append(image)

            if not +total_counter:
                break

        if +total_counter:
            raise RuntimeError(f"Couldn't find {self.config.n_shots} samples for some classes: {+total_counter}")

        sampled_images = [copy.deepcopy(x) for x in sampled_images]

        return DatasetManifest(sampled_images, copy.deepcopy(manifest.categories), copy.deepcopy(manifest.data_type), copy.deepcopy(manifest.additional_info))
