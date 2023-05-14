from ..resources.util import coco_dict_to_manifest


class BaseCocoAdaptor:
    def test_create_data_manifest(self, coco_dict):
        manifest = coco_dict_to_manifest(self.TASK, coco_dict)
        self.check(manifest, coco_dict)

    def check(self, manifest, coco_dict):
        assert len(manifest.images) == len(coco_dict['images'])
        categories = coco_dict.get('categories')
        if categories:
            assert manifest.categories and len(manifest.categories) == len(categories)
        assert sum([len(img.labels) for img in manifest.images]) == len(coco_dict['annotations'])
