from PIL import Image
import json
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Convert IRIS coco to AML coco format.')
    parser.add_argument('-i', '--images_txt_file', required=True, type=str, default='test_images.txt', help='image Text file name.')
    parser.add_argument('-l', '--label_name_file', required=True, type=str, default='labels.txt', help='label Text file name.')
    parser.add_argument('-o', '--output_file', required=True, type=str, default='AML_coco.json', help='output json name.')
    parser.add_argument('-u', '--blob_base_url', required=True, type=str, help='blob base url url for blob container.')

    return parser


def main():
    args = create_arg_parser().parse_args()
    logger.info(args.__dict__)

    aml_coco = {}
    images = []
    annotations = []
    categories = []
    image_id = 1
    annotations_id = 1
    categories_id = 1
    dimensions = {}

    if not os.path.exists(args.images_txt_file):
        logger.info(f'imageTextPath {args.images_txt_file} does not exist.')
        return

    if not os.path.exists(args.label_name_file):
        logger.info(f'labelTextPath {args.label_name_file} does not exist.')
        return

    with open(args.images_txt_file) as images_txt:
        for line in images_txt:
            image = {}

            image_path_zip, label_path_zip = line.strip().split(' ')
            image_path = '/'.join(image_path_zip.split('.zip@'))
            label_path = '/'.join(label_path_zip.split('.zip@'))

            with Image.open(image_path) as img:
                width = img.width
                height = img.height

            image['id'] = image_id
            image['width'] = width
            image['height'] = height
            image['file_name'] = image_path
            image['coco_url'] = f'{args.blob_base_url}/{image_path}'

            dimensions[image['id']] = (width, height)

            image_id += 1
            images.append(image)

            with open(label_path, 'r') as labels:
                for label in labels:
                    entry = label.strip().split(' ')
                    annotation = {}
                    annotation['id'] = annotations_id
                    annotations_id += 1
                    annotation['category_id'] = int(entry[0]) + 1
                    annotation['image_id'] = image['id']
                    annotation['area'] = abs(int(entry[1]) - int(entry[3])) * abs(int(entry[2]) - int(entry[4]))

                    if len(entry) > 1:
                        bbox = []
                        if float(entry[1]) > 1 or float(entry[2]) > 1 or float(entry[3]) > 1 or float(entry[4]) > 1:
                            bbox.append(int(entry[1])/int(dimensions[image['id']][0]))
                            bbox.append(int(entry[2])/int(dimensions[image['id']][1]))
                            bbox.append((int(entry[3]) - int(entry[1]))/int(dimensions[image['id']][0]))
                            bbox.append((int(entry[4]) - int(entry[2]))/int(dimensions[image['id']][1]))
                        else:
                            bbox.append(entry[1])
                            bbox.append(entry[2])
                            bbox.append(float(entry[3]) - float(entry[1]))
                            bbox.append(float(entry[4]) - float(entry[2]))

                        annotation['bbox'] = bbox

                    annotations.append(annotation)

    with open(args.label_name_file) as labels:
        for label in labels:
            category = {}
            category['id'] = categories_id
            category['name'] = label.strip()
            categories_id += 1
            categories.append(category)

    aml_coco['images'] = images
    aml_coco['annotations'] = annotations
    aml_coco['categories'] = categories

    with open(args.output_file, "w") as out_file:
        json.dump(aml_coco, out_file, indent=4)


if __name__ == '__main__':
    main()
