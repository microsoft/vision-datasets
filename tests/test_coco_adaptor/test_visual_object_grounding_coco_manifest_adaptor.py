import pytest
import copy
from vision_datasets.common import DatasetTypes
from .coco_adaptor_base import BaseCocoAdaptor
from ..resources.util import coco_database
from vision_datasets.visual_object_grounding import VisualObjectGroundingLabelManifest


class TestVisualObjectGrounding(BaseCocoAdaptor):
    TASK = DatasetTypes.VISUAL_OBJECT_GROUNDING

    @pytest.mark.parametrize("coco_dict", coco_database[TASK])
    def test_create_data_manifest(self, coco_dict):
        super().test_create_data_manifest(coco_dict)

    @pytest.mark.parametrize("coco_dict", coco_database[TASK])
    def test_create_data_manifest_with_additional_info(self, coco_dict):
        super().test_create_data_manifest_with_additional_info(coco_dict)

    @pytest.mark.parametrize("format", ["ltwh", "ltrb"])
    def test_bbox_format(self, format):
        coco_dict = copy.deepcopy(coco_database[self.TASK][0])
        coco_dict['bbox_format'] = format
        manifest = super().test_create_data_manifest(coco_dict)
        ann_by_image = [[] for _ in range(len(coco_dict['images']))]

        for ann in coco_dict['annotations']:
            ann_by_image[ann['image_id']-1].append(ann)

        for i, image in enumerate(manifest.images):
            image_anns = ann_by_image[i]
            for j, l in enumerate(image.labels):
                l: VisualObjectGroundingLabelManifest
                for k, g in enumerate(l.groundings):
                    for t, bbox in enumerate(g.bboxes):
                        gt = image_anns[j]["groundings"][k]['bboxes'][t]
                        gt_ltwh = gt if format == "ltwh" else [gt[0], gt[1], gt[2]-gt[0], gt[3]-gt[1]]
                        manifest_ltwh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                        assert gt_ltwh == manifest_ltwh
