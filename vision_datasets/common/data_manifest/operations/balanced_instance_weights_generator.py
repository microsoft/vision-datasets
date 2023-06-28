"""Generate instance weights from DatasetManifest, which can be used for balancing the dataset by sampling instances based on the weights. Only works for classification, detection, multitask."""

import logging
import typing
from collections import Counter
from dataclasses import dataclass

import numpy

from ..data_manifest import DatasetManifest, ImageLabelWithCategoryManifest
from .operation import Operation

logger = logging.getLogger(__name__)


@dataclass
class WeightsGenerationConfig:
    soft: bool = True  # less aggressive in making the dataset balanced
    weight_upper: float = 5.0
    weight_lower: float = 0.2


class BalancedInstanceWeightsGenerator(Operation):
    """
    Generate instance weights, with which sampling can achieve a balanced dataset across different categories.
    """
    _NEG_CLASS_INDEX = -1

    def __init__(self, config: WeightsGenerationConfig) -> None:
        super().__init__()
        self.config = config

    def run(self, *args: DatasetManifest):
        data_manifest = args[0]
        if data_manifest is None:
            raise ValueError('data manifest is None.')

        logger.info("Generating instance weights for dataset balancing.")
        image_tags = [self._process_labels(x.labels) for x in data_manifest.images]

        class_wise_image_counter = Counter()
        for tags in image_tags:
            class_wise_image_counter.update(tags)

        mean_class_wise_image_tag_count = numpy.mean(list(class_wise_image_counter.values()))
        class_wise_multipliers = {x: mean_class_wise_image_tag_count / class_wise_image_counter[x] for x in class_wise_image_counter}
        if self.config.soft:
            class_wise_multipliers = {x: numpy.sqrt(class_wise_multipliers[x]) for x in class_wise_multipliers}

        class_wise_multipliers = {x: BalancedInstanceWeightsGenerator._scope_multiplier(class_wise_multipliers[x], self.config.weight_upper, self.config.weight_lower) for x in class_wise_multipliers}

        image_weights = [BalancedInstanceWeightsGenerator._get_instance_multiplier(tags, class_wise_multipliers, self.config.weight_upper, self.config.weight_lower) for tags in image_tags]

        logger.info(f'instance weights: max {max(image_weights)}, min {min(image_weights)}, len {len(image_weights)}')

        return image_weights

    def _process_labels(self, labels: typing.List[ImageLabelWithCategoryManifest]):
        return [x.category_id for x in labels] or [BalancedInstanceWeightsGenerator._NEG_CLASS_INDEX]

    @staticmethod
    def _get_instance_multiplier(tags, class_wise_multipliers, weight_upper, weight_lower):
        mul = numpy.prod([class_wise_multipliers[tag] for tag in tags])

        return BalancedInstanceWeightsGenerator._scope_multiplier(mul, weight_upper, weight_lower)

    @staticmethod
    def _scope_multiplier(value, weight_upper, weight_lower):
        return min(max(value, weight_lower), weight_upper)
