"""
example inputs
images_txt_path = r'test_images.txt'
out_put_file = "marsb_regular_od_benchmark_20210904_AML_coco_test2.json"
label_name_path = r'labels.txt'
absolute_url = 'https://your_image_blob_path'
date_captured = '2022-06-24T23:34:17.7980622Z'
coco_url = 'AmlDatastore://marsb-regular/'
has_bbox = True
"""

from PIL import Image
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Convert IRIS coco to AML coco format.')
    parser.add_argument('-imageTextPath', '--images_txt_path', required=True, type=str, default='test_images.txt', help='image Text file name.')
    parser.add_argument('-labelTextPath', '--label_name_path', required=True, type=str, default='labels.txt', help='label Text file name.')
    parser.add_argument('-outputJsonName', '--out_put_file', required=True, type=str, default='AML_coco.json', help='output json name.')
    parser.add_argument('-absoluteUrl', '--absolute_url', required=True, type=str, default='https://your_image_blob_path', help='absolute url for blob container.')
    parser.add_argument('-hasBbox', '--has_bbox', required=True, type=bool, default=True, help='is object dection (has bounding box).')
    parser.add_argument('-projectUrl', '--coco_url', required=False, type=str, default='AmlDatastore://mobileone', help='project path (it is ok to keep default).')
    parser.add_argument('-dateCaptured', '--date_captured', required=False, type=str, default='2022-06-24T23:34:17.7980622Z', help='date captured (it is ok to keep default).')

    return parser


def main():
    args = create_arg_parser().parse_args()
    logger.info(args.__dict__)

    AML_coco = {}
    images = []
    annotations = []
    categories = []
    image_id = 1
    annotations_id = 1
    categories_id = 1
    dir_id_WH = {}

    with open(args.images_txt_path) as f:
        train_images = json.load(f)
        for line in train_images:
            image = {}
            WH = []

            image_path_zip, label_path_zip = line.strip().split(' ')
            image_path = '/'.join(image_path_zip.split('.zip@'))
            label_path = '/'.join(label_path_zip.split('.zip@'))

            img = Image.open(image_path)
            width = img.width
            height = img.height

            image['id'] = image_id
            image['width'] = width
            image['height'] = height
            image['file_name'] = image_path
            image['coco_url'] = args.coco_url + '/' + image_path
            image['absolute_url'] = args.absolute_url + '/' + image_path
            image['date_captured'] = args.date_captured

            WH.append(width)
            WH.append(height)
            dir_id_WH[image['id']] = WH
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

                    if args.has_bbox:
                        bbox = []
                        bbox.append(int(entry[1])/int(dir_id_WH[image['id']][0]))
                        bbox.append(int(entry[2])/int(dir_id_WH[image['id']][1]))
                        bbox.append((int(entry[3]) - int(entry[1]))/int(dir_id_WH[image['id']][0]))
                        bbox.append((int(entry[4]) - int(entry[2]))/int(dir_id_WH[image['id']][1]))
                        annotation['bbox'] = bbox

                    annotations.append(annotation)

    with open(args.label_name_path) as labels:
        for label in labels:
            category = {}
            category['id'] = categories_id
            category['name'] = label.strip()
            categories_id += 1
            categories.append(category)

    AML_coco['images'] = images
    AML_coco['annotations'] = annotations
    AML_coco['categories'] = categories

    with open(args.out_put_file, "w") as out_file:
        json.dump(AML_coco, out_file, indent=4)


if __name__ == '__main__':
    main()
