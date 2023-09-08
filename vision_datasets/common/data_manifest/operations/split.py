import random
import typing
from copy import deepcopy
from dataclasses import dataclass

from ..data_manifest import DatasetManifest
from .operation import Operation


@dataclass
class SplitConfig:
    ratio: float
    random_seed: int = 0


class Split(Operation):
    """
        Split the dataset into two sets.
        For multiclass dataset, the split ratio will be close to provided ratio, while for multilabel dataset, it is not guaranteed
        Multitask dataset and detection dataset are treated the same with multilabel dataset.
    """

    def __init__(self, config: SplitConfig) -> None:
        super().__init__()
        self.config = config

    def run(self, *args: DatasetManifest):
        if len(args) != 1:
            raise ValueError

        manifest = args[0]
        first_cnt = int(self.config.ratio * len(manifest))
        if first_cnt == 0:
            return DatasetManifest([], deepcopy(manifest.categories), deepcopy(manifest.data_type), deepcopy(manifest.additional_info)), \
                DatasetManifest(deepcopy(manifest.images), deepcopy(manifest.categories), deepcopy(manifest.data_type), deepcopy(manifest.additional_info))

        if first_cnt == len(manifest):
            return DatasetManifest(deepcopy(manifest.images), deepcopy(manifest.categories), deepcopy(manifest.data_type), deepcopy(manifest.additional_info)), \
                DatasetManifest([], deepcopy(manifest.categories), deepcopy(manifest.data_type), deepcopy(manifest.additional_info))

        rng = random.Random(self.config.random_seed)
        images = deepcopy(manifest.images)
        rng.shuffle(images)

        return DatasetManifest(images[: first_cnt], deepcopy(manifest.categories), deepcopy(manifest.data_type), deepcopy(manifest.additional_info)), \
            DatasetManifest(images[first_cnt:], deepcopy(manifest.categories), deepcopy(manifest.data_type), deepcopy(manifest.additional_info))


class SplitWithCategories(Operation):
    def __init__(self, config: SplitConfig) -> None:
        super().__init__()
        self.config = config

    def run(self, *args: DatasetManifest):
        if len(args) != 1:
            raise ValueError

        manifest = args[0]
        if int(len(manifest.images) * self.config.ratio) == 0:
            return DatasetManifest(
                [],
                deepcopy(manifest.categories),
                deepcopy(manifest.data_type),
                deepcopy(manifest.additional_info)), DatasetManifest(
                deepcopy(manifest.images),
                deepcopy(manifest.categories),
                deepcopy(manifest.data_type),
                deepcopy(manifest.additional_info))

        if int(len(manifest.images) * self.config.ratio) == len(manifest.images):
            return DatasetManifest(
                deepcopy(manifest.images),
                deepcopy(manifest.categories),
                deepcopy(manifest.data_type),
                deepcopy(manifest.additional_info)), DatasetManifest(
                [],
                deepcopy(manifest.categories),
                deepcopy(manifest.data_type),
                deepcopy(manifest.additional_info))

        rng = random.Random(self.config.random_seed)
        images = deepcopy(manifest.images)
        rng.shuffle(images)

        first_imgs = []
        second_imgs = []
        n_first_imgs_by_class = [0] * len(manifest.categories)
        n_second_imgs_by_class = [0] * len(manifest.categories)
        first_to_second_ratio = (1 - self.config.ratio) / self.config.ratio
        n_first_negative_imgs = 0
        n_second_negative_imgs = 0

        def get_img_label_cnt(labels, n_images_by_class: typing.List) -> typing.List[int]:
            return [n_images_by_class[label.category_id] for label in labels]

        def add_cnt(labels, n_images_by_class: typing.List):
            for label in labels:
                n_images_by_class[label.category_id] += 1

        for image in images:
            if image.is_negative():
                if n_first_negative_imgs == 0 or n_second_negative_imgs / n_first_negative_imgs >= first_to_second_ratio:
                    n_first_negative_imgs += 1
                    first_imgs.append(image)
                else:
                    n_second_negative_imgs += 1
                    second_imgs.append(image)

                continue

            img_label_cnt_in_first = get_img_label_cnt(image.labels, n_first_imgs_by_class)
            img_label_cnt_in_second = get_img_label_cnt(image.labels, n_second_imgs_by_class)
            first_cnt_sum = sum(img_label_cnt_in_first) * first_to_second_ratio
            first_cnt_min = min(img_label_cnt_in_first) * first_to_second_ratio
            second_cnt_sum = sum(img_label_cnt_in_second)
            second_cnt_min = min(img_label_cnt_in_second)
            if second_cnt_min < first_cnt_min or (second_cnt_min == first_cnt_min and second_cnt_sum < first_cnt_sum):
                second_imgs.append(image)
                add_cnt(image.labels, n_second_imgs_by_class)
            else:
                first_imgs.append(image)
                add_cnt(image.labels, n_first_imgs_by_class)

        return DatasetManifest(first_imgs, deepcopy(manifest.categories), deepcopy(manifest.data_type), deepcopy(manifest.additional_info)), \
            DatasetManifest(second_imgs, deepcopy(manifest.categories), deepcopy(manifest.data_type), deepcopy(manifest.additional_info))
