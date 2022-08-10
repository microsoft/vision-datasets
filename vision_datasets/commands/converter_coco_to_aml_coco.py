import json
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Convert IRIS coco to AML coco format.')
    parser.add_argument('-i', '--standard_coco_file', required=True, type=str, help='Standard coco file to convert.')
    parser.add_argument('-o', '--output_file', required=True, type=str, default='AML_coco.json', help='output coco file name.')
    parser.add_argument('-u', '--blob_base_url', required=True, type=str, help='blob base url for blob container.')

    return parser


def main():
    args = create_arg_parser().parse_args()
    logger.info(args.__dict__)

    aml_images = []
    aml_annotations = []
    dimensions = {}

    if not os.path.exists(args.standard_coco_file):
        logger.info(f'inputCocoName {args.standard_coco_file} does not exist.')
        return

    with open(args.standard_coco_file) as f:
        iris_file = json.load(f)

        images = iris_file['images']
        annotations = iris_file["annotations"]
        categories = iris_file['categories']

        for image in images:
            image_tmp = {}
            file_name = image['file_name'].replace('.zip@', '/')

            image_tmp['id'] = image['id']
            image_tmp['width'] = image['width']
            image_tmp['height'] = image['height']
            image_tmp['file_name'] = file_name
            image_tmp['absolute_url'] = f'{args.blob_base_url}/{file_name}'

            dimensions[image['id']] = (image['width'], image['height'])

            aml_images.append(image_tmp)

        for annotation in annotations:
            annotation_tmp = {}
            annotation_tmp['id'] = annotation['id']
            annotation_tmp['category_id'] = annotation['category_id']
            annotation_tmp['image_id'] = annotation['image_id']
            annotation_tmp['area'] = annotation['area']

            if 'bbox' in annotation:
                bbox = []
                if annotation['bbox'][0] > 1 or annotation['bbox'][1] > 1 or annotation['bbox'][2] > 1 or annotation['bbox'][3] > 1:
                    bbox.append(annotation['bbox'][0]/dimensions[annotation['image_id']][0])
                    bbox.append(annotation['bbox'][1]/dimensions[annotation['image_id']][1])
                    bbox.append(annotation['bbox'][2]/dimensions[annotation['image_id']][0])
                    bbox.append(annotation['bbox'][3]/dimensions[annotation['image_id']][1])
                    annotation_tmp['bbox'] = bbox
                else:
                    annotation_tmp['bbox'] = annotation['bbox'][0]

            aml_annotations.append(annotation_tmp)

    aml_coco = {}
    aml_coco['images'] = aml_images
    aml_coco['annotations'] = aml_annotations
    aml_coco['categories'] = categories

    with open(args.output_file, "w") as out_file:
        json.dump(aml_coco, out_file, indent=4)


if __name__ == '__main__':
    main()
