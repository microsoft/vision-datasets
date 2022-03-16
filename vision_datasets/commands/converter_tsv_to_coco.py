"""
Converts tsv format to coco format
ic: img_id  class_id    img_data_base64
od: img_id  [{"class": class_name, "rect": [L, T, R, B], "diff": 0}, ...] img_data_base64
"""

import logging
import json
import os

from .utils import verify_and_correct_box_or_none, guess_encoding, decode64_to_pil, zip_folder, TSV_FORMAT_LTRB, TSV_FORMAT_LTWH_NORM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Convert tsv data to coco format.')
    parser.add_argument('-t', '--tsvs', required=True, nargs='+', help='Tsv files to convert.')
    parser.add_argument('-c', '--task', required=True, type=str, help='type of tasks.', choices=['ic', 'od'])
    parser.add_argument('-l', '--labelmap', type=str, default='labelmap.txt')
    parser.add_argument('-f', '--format', type=str, default=TSV_FORMAT_LTRB, choices=[TSV_FORMAT_LTRB, TSV_FORMAT_LTWH_NORM])
    parser.add_argument('-d', '--difficulty', type=bool, default=False, help='Include difficulty boxes or not.')
    parser.add_argument('-z', '--zip', type=bool, default=False, help='Zip the image and label folders or not.')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Output folder for images.')

    return parser


def main():
    args = create_arg_parser().parse_args()
    logger.info(args.__dict__)
    image_folder_name = args.output_folder

    if not os.path.exists(image_folder_name):
        os.mkdir(image_folder_name)

    labelmap_exists = os.path.exists(args.labelmap)
    if labelmap_exists:
        logger.info(f'Labelmap {args.labelmap} exists.')
        with open(args.labelmap, 'r') as labelmap_in:
            categories = [x.strip() for x in labelmap_in.readlines()]
            if not categories:
                raise Exception(f'Empty labelmap {args.labelmap}')
            label_name_to_idx = {x: i for i, x in enumerate(categories)}
            categories = [{'name': x, 'id': i + 1} for i, x in enumerate(categories)]
    else:
        assert args.task != 'ic', 'labelmap must exist for ic.'
        logger.info(f'Labelmap {args.labelmap} does not exist, created on the fly.')
        label_name_to_idx = dict()
        categories = None

    for tsv_file_name in args.tsvs:
        line_idx = 1
        images = []
        annotations = []
        with open(tsv_file_name, 'r', encoding=guess_encoding(tsv_file_name)) as file_in:
            logger.info(f'Processing {tsv_file_name}.')
            for img_info in file_in:
                img_id, labels, img_b64 = img_info.split('\t')

                img = decode64_to_pil(img_b64)
                w, h = img.size

                # image data => image file
                img_file_name = img_id.replace('/', '_') + '.' + img.format
                img.save(os.path.join(image_folder_name, img_file_name), img.format)

                image_path = f'{image_folder_name}.zip@{img_file_name}'

                img_info_dict = {'id': line_idx, 'width': w, 'height': h, 'file_name': image_path}
                images.append(img_info_dict)

                # image info => index file

                lp = f'File: {tsv_file_name}, Line {line_idx}: '

                if args.task == 'ic':
                    for label_idx in labels.split(','):
                        annotations.append({'id': len(annotations) + 1, 'category_id': int(label_idx) + 1, 'image_id': line_idx})
                else:
                    # labels => files
                    labels = json.loads(labels)
                    for label in labels:
                        difficulty = label['diff'] if 'diff' in label else 0
                        if difficulty > 0 and not args.difficulty:
                            continue

                        if label['class'] not in label_name_to_idx:
                            if labelmap_exists:
                                raise Exception(f'{lp}Illegal class {label["class"]}, not in provided labelmap.')
                            else:
                                label_name_to_idx[label['class']] = len(label_name_to_idx)

                        label_idx = label_name_to_idx[label['class']]
                        box = verify_and_correct_box_or_none(lp, label['rect'], args.format, w, h)
                        if box is None:
                            continue

                        annotations.append({'id': len(annotations) + 1, 'category_id': label_idx + 1, 'image_id': line_idx, 'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]]})

                line_idx += 1
                if line_idx % 2000 == 0:
                    logger.info(f'Processed {line_idx} images.')

        categories = categories or [{'id': idx + 1, 'name': name} for name, idx in label_name_to_idx.items()]
        coco_file_name = os.path.splitext(os.path.basename(tsv_file_name))[0] + '.json'
        with open(coco_file_name, 'w') as coco_out:
            coco_out.write(json.dumps({'images': images, 'annotations': annotations, 'categories': categories}, indent=2))

    if not labelmap_exists:
        logger.info(f'Write labelmap to {args.labelmap}')
        with open(args.labelmap, 'w') as labelmap_out:
            idx_to_labels = {label_name_to_idx[key]: key for key in label_name_to_idx}
            for i in range(len(idx_to_labels)):
                labelmap_out.write(idx_to_labels[i] + '\n')

    if args.zip:
        logger.info(f'Zip folder "{image_folder_name}".')
        zip_folder(image_folder_name)


if __name__ == '__main__':
    main()
