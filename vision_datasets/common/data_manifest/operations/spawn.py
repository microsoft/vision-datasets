import copy
import logging
import typing
from dataclasses import dataclass

from ..data_manifest import DatasetManifest
from .merge import SingleTaskMerge
from .operation import Operation
from .sample import SampleByNumSamples, SampleByNumSamplesConfig

logger = logging.getLogger(__name__)


@dataclass
class SpawnConfig:
    random_seed: int
    target_n_samples: int
    instance_weights: typing.List[float] = None


class Spawn(Operation):
    """
    Spawn the dataset (oversample).

    This will be consolidated with sample operation.
    """

    def __init__(self, config: SpawnConfig) -> None:
        super().__init__()
        self.config = config

    def run(self, *args: DatasetManifest):
        """Spawn manifest to a size.
        To ensure each class has samples after spawn, we first keep a copy of original data, then merge with sampled data.
        If instance_weights is not provided, spawn follows class distribution.
        Otherwise spawn the dataset so that the instances follow the given weights. In this case the spawned size is not guranteed to be num_samples.

        Returns:
            Spawned dataset (DatasetManifest)
        """

        if len(args) != 1:
            raise ValueError

        manifest = args[0]
        cfg = self.config
        if cfg.instance_weights:
            if len(cfg.instance_weights) != len(manifest) or any([x < 0 for x in cfg.instance_weights]):
                raise ValueError

            sum_weights = sum(cfg.instance_weights)
            # Distribute the number of num_samples to each image by the weights. The original image is subtracted.
            n_copies_per_sample = [max(0, round(w / sum_weights * cfg.target_n_samples - 1)) for w in cfg.instance_weights]
            spawned_images = []
            for image, n_copies in zip(manifest.images, n_copies_per_sample):
                spawned_images += [copy.deepcopy(image) for _ in range(n_copies)]

            sampled_manifest = DatasetManifest(spawned_images, manifest.categories, manifest.data_type, manifest.additional_info)
        else:
            cfg = SampleByNumSamplesConfig(cfg.random_seed, True, cfg.target_n_samples - len(manifest))
            sampled_manifest = SampleByNumSamples(cfg).sample(manifest)

        # Merge with the copy of the original dataset to ensure each class has sample.
        merger = SingleTaskMerge()
        return merger.merge(manifest, sampled_manifest)
