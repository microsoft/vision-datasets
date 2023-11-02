import argparse
import importlib
import io
import json
import locale
import logging
import os
import pathlib
import zipfile
from typing import Union

from tqdm import tqdm

from vision_datasets import DatasetManifest, DatasetTypes, Usages
from vision_datasets.common import Base64Utils, StandAloneImageListGeneratorFactory


def set_up_cmd_logger(name):
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO)

    return logger


logger = set_up_cmd_logger(__name__)

TSV_FORMAT_LTRB = 'ltrb'
TSV_FORMAT_LTWH_NORM = 'ltwh-normalized'


def enum_type(enum_type):
    def func(value_str):
        try:
            return enum_type[value_str.upper()]
        except KeyError:
            raise argparse.ArgumentTypeError(f"'{value_str}' is not a valid value of {value_str}. Choose from: {[e.name for e in enum_type]}")

    return func


def add_args_to_locate_dataset_from_name_and_reg_json(parser):
    parser.add_argument('name', type=str, help='Dataset name.')
    parser.add_argument('--reg_json', '-r', type=pathlib.Path, default=None, help='dataset registration json file path.', required=False)
    parser.add_argument('--version', '-v', type=int, help='Dataset version.', default=None)
    parser.add_argument('--usages', '-u', nargs='+', choices=list(Usages), type=enum_type(Usages), default=[Usages.TRAIN, Usages.VAL, Usages.TEST], help='Usage(s) to check.')

    parser.add_argument('--blob_container', '-k', type=str, help='Blob container (sas) url', required=False)
    parser.add_argument('--local_dir', '-f', type=pathlib.Path, required=False, help='Check the dataset in this folder. Folder will be created if not exist and blob_container is provided.')


def add_args_to_locate_dataset(parser):
    add_args_to_locate_dataset_from_name_and_reg_json(parser)

    parser.add_argument('--coco_json', '-c', type=pathlib.Path, default=None, help='Single coco json file to check.', required=False)
    parser.add_argument('--data_type', '-t', type=enum_type(DatasetTypes), default=None, help='Type of data.', choices=list(DatasetTypes), required=False)


def get_or_generate_data_reg_json_and_usages(args):
    def _generate_reg_json(name, type, coco_path):
        data_info = [
            {
                'name': name,
                'version': 1,
                'type': type,
                'format': 'coco',
                'root_folder': '',
                'train': {
                    'index_path': coco_path.name
                }
            }
        ]

        return json.dumps(data_info)

    if args.reg_json:
        usages = args.usages or [Usages.TRAIN, Usages.VAL, Usages.TEST]
        data_reg_json = args.reg_json.read_text()
    else:
        assert args.coco_json, '--coco_json not provided'
        assert args.data_type, '--data_type not provided'
        usages = [Usages.TRAIN]
        data_reg_json = _generate_reg_json(args.name, args.data_type, args.coco_json)

    return data_reg_json, usages


def zip_folder(folder_name, direct=False):
    zip_file = zipfile.ZipFile(f'{folder_name}.zip', 'w', zipfile.ZIP_STORED)
    i = 0
    for root, dirs, files in tqdm(os.walk(folder_name), desc=f'Zipping {folder_name}...'):
        for file in files:
            if i and i % 1000 == 0:
                logger.info(f'Zipped {i} images..')

            if direct:
                zip_file.write(os.path.join(root, file), pathlib.Path(pathlib.Path(root).name) / file)
            else:
                zip_file.write(os.path.join(root, file))
            i += 1

    logger.info(f'Zipped {i} images in total.')
    zip_file.close()


def generate_reg_json(name, type, coco_path):
    data_info = [
        {
            'name': name,
            'version': 1,
            'type': type,
            'format': 'coco',
            'root_folder': '',
            'train': {
                'index_path': coco_path.name
            }
        }
    ]

    return json.dumps(data_info)


def convert_to_tsv(manifest: DatasetManifest, file_path: Union[str, pathlib.Path]):
    with open(file_path, 'w', encoding='utf-8') as file_out:
        for img in tqdm(manifest.images, desc=f'Writing to {file_path}'):
            converted_labels = []
            for label in img.labels:
                if manifest.data_type in [DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS]:
                    tag_name = manifest.categories[label]
                    converted_label = {'class': tag_name}
                elif manifest.data_type == DatasetTypes.IMAGE_OBJECT_DETECTION:
                    tag_name = manifest.categories[label[0]]
                    rect = [int(x) for x in label[1:5]]

                    # to LTRB format
                    converted_label = {'class': tag_name, 'rect': rect}
                elif manifest.data_type == DatasetTypes.IMAGE_CAPTION:
                    converted_label = {'caption': label}

                converted_labels.append(converted_label)

            b64img = Base64Utils.file_to_b64_str(pathlib.Path(img.img_path))
            file_out.write(f'{img.id}\t{json.dumps(converted_labels, ensure_ascii=False)}\t{b64img}\n')


def convert_to_jsonl(manifest: DatasetManifest, file_path: Union[str, pathlib.Path], flatten=True):
    generator = StandAloneImageListGeneratorFactory.create(manifest.data_type, flatten=flatten)
    with open(file_path, 'w', encoding='utf-8') as file_out:
        for item in tqdm(generator.run(manifest), desc=f'Writing to {file_path}.'):
            file_out.write(json.dumps(item, ensure_ascii=False) + '\n')


def guess_encoding(tsv_file):
    """guess the encoding of the given file https://stackoverflow.com/a/33981557/
    """
    assert tsv_file

    with io.open(tsv_file, 'rb') as f:
        data = f.read(5)
    if data.startswith(b'\xEF\xBB\xBF'):  # UTF-8 with a "BOM"
        return 'utf-8-sig'
    elif data.startswith(b'\xFF\xFE') or data.startswith(b"\xFE\xFF"):
        return 'utf-16'
    else:  # in Windows, guessing utf-8 doesn't work, so we have to try
        # noinspection PyBroadException
        try:
            with io.open(tsv_file, encoding='utf-8') as f:
                f.read(222222)
                return 'utf-8'
        except Exception:
            return locale.getdefaultlocale()[1]


def verify_and_correct_box_or_none(lp, box, data_format, img_w, img_h):
    error_msg = f'{lp} Illegal box [{", ".join([str(x) for x in box])}], img wxh: {img_w}, {img_h}'
    if len([x for x in box if x < 0]) > 0:
        logger.error(f'{error_msg}. Skip this box.')
        return None

    if data_format == TSV_FORMAT_LTWH_NORM:
        box[2] = int((box[0] + box[2]) * img_w)
        box[3] = int((box[1] + box[3]) * img_h)
        box[0] = int(box[0] * img_w)
        box[1] = int(box[1] * img_h)

    boundary_ratio_limit = 1.02
    if box[0] >= img_w or box[1] >= img_h or box[2] / img_w > boundary_ratio_limit \
            or box[3] / img_h > boundary_ratio_limit or box[0] >= box[2] or box[1] >= box[3]:
        logger.error(f'{error_msg}. Skip this box.')
        return None

    box[2] = min(box[2], img_w)
    box[3] = min(box[3], img_h)

    return box


def write_to_json_file_utf8(dict, filepath: Union[str, pathlib.Path]):
    assert filepath

    pathlib.Path(filepath).write_text(json.dumps(dict, indent=2, ensure_ascii=False), encoding='utf-8')


def is_module_available(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ModuleNotFoundError:
        return False
