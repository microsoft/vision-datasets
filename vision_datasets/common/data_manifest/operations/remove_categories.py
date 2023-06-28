import copy
import typing
from dataclasses import dataclass

from ..data_manifest import DatasetManifest
from .operation import Operation


@dataclass
class RemoveCategoriesConfig:
    category_names: typing.List[str]


class RemoveCategories(Operation):
    """
    Remove categories.
    """
    def __init__(self, config: RemoveCategoriesConfig) -> None:
        super().__init__()
        if not config:
            raise ValueError

        self.config = config

    def run(self, *args: DatasetManifest):
        if len(args) != 1:
            raise ValueError

        manifest = args[0]
        if not manifest.categories:
            raise ValueError

        result = copy.deepcopy(manifest)
        if not self.config.category_names:
            return result

        c_name_to_idx = {c.name: i for i, c in enumerate(manifest.categories)}
        c_indices_to_remove = sorted([c_name_to_idx[c] for c in self.config.category_names])
        old_c_idx_to_new_idx = {}
        j = 0
        for i in range(len(manifest.categories)):
            if j < len(c_indices_to_remove) and i == c_indices_to_remove[j]:
                j += 1
            else:
                old_c_idx_to_new_idx[i] = i - j

        def alter_cid(label, new_category_id):
            label.category_id = new_category_id
            return label

        for image in result.images:
            image.labels = [alter_cid(label, old_c_idx_to_new_idx[label.category_id]) for label in image.labels if label.category_id in old_c_idx_to_new_idx]

        def alter_category(category, new_idx):
            category.id = new_idx
            return category

        result.categories = [alter_category(c, old_c_idx_to_new_idx[i]) for i, c in enumerate(result.categories) if i in old_c_idx_to_new_idx]
        return result
