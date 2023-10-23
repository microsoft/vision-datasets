import argparse
import json
import pathlib
from enum import Enum
from PIL import Image
from .utils import add_args_to_locate_dataset, enum_type, set_up_cmd_logger, get_or_generate_data_reg_json_and_usages
from vision_datasets import DatasetHub, DatasetTypes
from vision_datasets.common import CocoDictGeneratorFactory

logger = set_up_cmd_logger(__name__)


def logging_prefix(dataset_name, version):
    return f'Modify images in dataset {dataset_name}, version {version}: '


class Format(Enum):
    JPG = 1
    PNG = 2
    BMP = 3
    TIFF = 4


def resize_image_by_longer_edge(image: Image.Image, new_size):
    width, height = image.size
    is_width_shorter = width < height

    # Calculate new dimensions
    if is_width_shorter:
        new_width = new_size
        new_height = int((new_size / width) * height)
    else:
        new_height = new_size
        new_width = int((new_size / height) * width)

    # Resize the image
    resized_img = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_img


def rotate_image(image: Image.Image, angle):
    rotated_img = image.rotate(angle)
    return rotated_img


def process_and_save_image(image: Image.Image, longer_edge_size, rotate_angle, format: Format, target_path: pathlib.Path):
    if longer_edge_size != None:
        image = resize_image_by_longer_edge(image, longer_edge_size)

    if rotate_angle != None:
        image = rotate_angle(image, rotate_angle)

    if format == Format.JPG:
        image.save(target_path.open(), quality=100, format='JPEG')
    elif format == Format.PNG:
        image.save(target_path.open(), compress_level=0, format='PNG')
    else:
        image.save(target_path.open(), format=str(format))

    return image


def main():
    parser = argparse.ArgumentParser('Alter images from a dataset.')
    add_args_to_locate_dataset(parser)
    parser.add_argument('--format', '-f', choices=list(Format), type=enum_type(Format), default=None, help='Format of image to be converted to.')
    parser.add_argument('--longer-edge-size', '-s', type=int, default=None, help='Image size of longer edge.')
    parser.add_argument('--rotate-angle', '-r', type=int, default=None, help='Rotate angle.')
    parser.add_argument('-o', '--output_folder', type=pathlib.Path, required=True, help='target folder of the converted classification dataset')

    args = parser.parse_args()
    prefix = logging_prefix(args.name, args.version)

    data_reg_json, usages = get_or_generate_data_reg_json_and_usages(args)
    dataset_hub = DatasetHub(data_reg_json, args.blob_container, args.local_dir.as_posix())
    dataset_info = dataset_hub.dataset_registry.get_dataset_info(args.name, args.version)

    if dataset_info.type in [DatasetTypes.IMAGE_OBJECT_DETECTION, DatasetTypes.VISUAL_OBJECT_GROUNDING]:
        raise ValueError(f'Dataset type {dataset_info.type} not supported.')

    if not dataset_info:
        logger.error(f'{prefix} dataset does not exist.')
        return

    if args.blob_container and args.local_dir:
        args.local_dir.mkdir(parents=True, exist_ok=True)

    coco_geneerator = CocoDictGeneratorFactory.create(dataset_info.type)
    for usage in usages:
        logger.info(f'{prefix} modify images in dataset with usage: {usage}.')

        # if args.local_dir is none, then this check will directly try to access data from azure blob. Images must be present in uncompressed folder on azure blob.
        dataset = dataset_hub.create_vision_dataset(name=dataset_info.name, version=args.version, usage=usage, coordinates='absolute')
        if dataset:
            manifest = dataset.dataset_manifest
            base_dir: pathlib.Path = args.output_folder / f'{usage}'
            base_dir.mkdir(parents=True, exist_ok=True)
            for i, sample in enumerate(dataset):
                img, _, _ = sample
                file_path = f'{i}.{args.format or img.format}'
                img = process_and_save_image(img, args.longer_edge_size, args.rotate_angle, args.format, base_dir / file_path)
                manifest.images[i].img_path = file_path

            coco_dict = coco_geneerator.run(manifest)
            pathlib.Path(f'{usage}.json').write_text(json.dumps(coco_dict, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
