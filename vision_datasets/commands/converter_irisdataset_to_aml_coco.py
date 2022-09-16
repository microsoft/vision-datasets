import json
import logging
from vision_datasets import DatasetHub
from vision_datasets.common.constants import DatasetTypes
import pathlib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Convert dataset to AML coco format.')
    parser.add_argument('-r', '--dataset_reg_file', required=True, type=pathlib.Path, help='dataset registration json path.')
    parser.add_argument('-l', '--local_dir', required=True, type=pathlib.Path, help='local directory for storage.')
    parser.add_argument('-o', '--output_file', required=True, type=str, default='aml_coco.json', help='output aml coco file name.')
    parser.add_argument('-b', '--blob_container_sas', required=True, type=str, help='blob base url url for blob container.')
    parser.add_argument('-n', '--dataset_name', required=True, type=str, help='dataset name.')

    return parser


def main():
    args = create_arg_parser().parse_args()
    logger.info(args.__dict__)

    dataset_resources = DatasetHub(pathlib.Path(args.dataset_reg_file).read_text())
    dataset = dataset_resources.create_manifest_dataset(args.blob_container_sas, args.local_dir, args.dataset_name, version=1, usage='train')

    if not dataset:
        logger.info(f'Skipping non-existent dataset_reg_file {args.dataset_reg_file}.')
        return

    aml_coco = {}
    images = []
    annotations = []
    categories = []
    image_id = 1
    annotations_id = 1
    categories_id = 1

    for img_man in dataset.images:
        image = {}
        image['id'] = image_id
        image['width'] = img_man.width
        image['height'] = img_man.height
        image['file_name'] = img_man.img_path
        image['coco_url'] = f'{args.blob_base_url}/{img_man.img_path}'

        images.append(image)

        for ann in img_man.labels:
            coco_ann = {
                'id': annotations_id,
                'image_id': image_id,
            }
            annotations_id += 1
            if DatasetTypes.is_classification(img_man.data_type):
                coco_ann['category_id'] = ann + 1
            elif img_man.data_type == DatasetTypes.OD:
                coco_ann['category_id'] = ann[0] + 1
                coco_ann['area'] = abs(ann[1] - ann[3]) * abs(ann[2] - ann[4])
                if ann[1] > 1 or ann[2] > 1 or ann[3] > 1 or ann[4] > 1:
                    coco_ann['bbox'] = [ann[1], ann[2], ann[3] - ann[1], ann[4] - ann[2]]
                else:
                    coco_ann['bbox'] = [ann[1]/img_man.width, ann[2]/img_man.height, (ann[3] - ann[1])/img_man.width, (ann[4] - ann[2])/img_man.height]
            elif img_man.data_type == DatasetTypes.IMCAP:
                coco_ann['caption'] = ann
            else:
                raise ValueError(f'Unsupported data type {img_man.data_type}')

            annotations.append(coco_ann)

        image_id += 1

    for label in dataset.labelmap:
        category = {}
        category['id'] = categories_id
        category['name'] = label
        categories_id += 1
        categories.append(category)

    aml_coco['images'] = images
    aml_coco['annotations'] = annotations
    aml_coco['categories'] = categories

    with open(args.output_file, "w") as out_file:
        json.dump(aml_coco, out_file, indent=4)


if __name__ == '__main__':
    main()
