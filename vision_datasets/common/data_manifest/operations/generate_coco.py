import abc

from ..data_manifest import DatasetManifest
from .operation import Operation


class GenerateCocoDictBase(Operation):
    """
    Base class for generating a COCO dictionary from DatasetManifest that can be serialized
    """

    def _generate_annotations(self, manifest: DatasetManifest):
        annotations = []
        for img_id, img in enumerate(manifest.images):
            for ann in img.labels:
                coco_ann = {
                    'id': len(annotations) + 1,
                    'image_id': img_id + 1,
                }

                self.process_labels(coco_ann, ann)
                annotations.append(coco_ann)

        return annotations

    def _generate_images(self, manifest):
        images = [{'id': i + 1, 'file_name': x.img_path, 'width': x.width, 'height': x.height} for i, x in enumerate(manifest.images)]
        return images

    def run(self, *args):
        if len(args) != 1:
            raise ValueError

        manifest = args[0]
        result = {
            "images": self._generate_images(manifest),
            "categories": self.generate_categories_or_none(manifest),
            "annotations": self._generate_annotations(manifest)
        }

        GenerateCocoDictBase._filter_none(result)
        return result

    @abc.abstractmethod
    def process_labels(self, coco_ann, label):
        pass

    def generate_categories_or_none(self, manifest):
        if manifest.categories:
            return [{'id': i + 1, 'name': x.name, 'supercateogry': x.super_category} for i, x in enumerate(manifest.categories)]

        return None

    @staticmethod
    def _filter_none(dict_val: dict):
        to_del = []
        for key in dict_val.keys():
            if dict_val[key] is None:
                to_del.append(key)
            elif isinstance(dict_val[key], dict):
                GenerateCocoDictBase._filter_none(dict_val[key])
            elif isinstance(dict_val[key], list):
                for x in dict_val[key]:
                    if isinstance(x, dict):
                        GenerateCocoDictBase._filter_none(x)

        for key in to_del:
            del dict_val[key]
