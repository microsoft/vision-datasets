import argparse
import json
import math
import numpy as np
import pathlib
import random
from enum import Enum

from PIL import Image
from tqdm import tqdm

from vision_datasets import DatasetHub, DatasetTypes
from vision_datasets.common import CocoDictGeneratorFactory

from .utils import add_args_to_locate_dataset, enum_type, get_or_generate_data_reg_json_and_usages, set_up_cmd_logger, zip_folder

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
    is_width_longer = width > height

    # Calculate new dimensions
    if is_width_longer:
        new_width = new_size
        new_height = int((new_size / width) * height)
    else:
        new_height = new_size
        new_width = int((new_size / height) * width)

    # Resize the image
    resized_img = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_img


def rotate_image(image, angle):
    """
    Rotate the image by a specified angle and ensure all pixels are retained.
    """
    # Convert angle to radians
    angle_rad = math.radians(angle)

    # Calculate the size of the new bounding box
    w, h = image.size
    new_width = int(math.ceil(abs(w * math.cos(angle_rad)) + abs(h * math.sin(angle_rad))))
    new_height = int(math.ceil(abs(h * math.cos(angle_rad)) + abs(w * math.sin(angle_rad))))

    # Create a new image with the calculated width and height
    new_image = Image.new("RGB", (new_width, new_height))

    # Paste the original image onto the center of the new image
    new_image.paste(image, ((new_width - w) // 2, (new_height - h) // 2))

    # Rotate the image
    rotated_image = new_image.rotate(-angle)

    return rotated_image


def process_and_save_image(image: Image.Image, longer_edge_size, rotate_angle, format: str, target_path: pathlib.Path):
    assert format

    if longer_edge_size is not None:
        image = resize_image_by_longer_edge(image, longer_edge_size)

    if rotate_angle is not None:
        image = rotate_image(image, rotate_angle)

    if format == Format.JPG:
        image.save(target_path, quality=100, format='JPEG')
    elif format == Format.PNG:
        image.save(target_path, compress_level=0, format='PNG')
    else:
        image.save(target_path, format=str(format))

    return image


def log_hist(name, vals, n_bins=5):
    hist, bin_edges = np.histogram(vals, bins=n_bins)

    for i in range(n_bins):
        logger.info(f"{name} Bin {i+1}: Range ({bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}), Count: {hist[i]}")


def main():
    parser = argparse.ArgumentParser('Alter images from a dataset.')
    add_args_to_locate_dataset(parser)
    parser.add_argument('--format', '-ft', choices=list(Format), type=enum_type(Format), default=None, help='Format of image to be converted to.')
    parser.add_argument('--longer-edge-size', '-es', type=str, default=None, help='Image size of longer edge. Two ints (sep by comma) for random size in a range; One int for a fixed size.')
    parser.add_argument('--rotate-angle', '-ra', type=str, default=None, help='Rotate angle. Two ints (sep by comma) for random angels in a range; One int for a fixed angle.')
    parser.add_argument('--output_folder', '-o', type=pathlib.Path, required=True, help='target folder of the converted classification dataset')
    parser.add_argument('--rnd_seed', '-rs', type=int, required=False, help='random see.', default=0)
    parser.add_argument('--zip', '-z', action='store_true', default=False, help='Zip the image and label folders or not.')

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

    coco_generator = CocoDictGeneratorFactory.create(dataset_info.type)
    random.seed(args.rnd_seed)
    for usage in usages:
        logger.info(f'{prefix} modify images in dataset with usage: {usage}.')

        # if args.local_dir is none, then this check will directly try to access data from azure blob. Images must be present in uncompressed folder on azure blob.
        dataset = dataset_hub.create_vision_dataset(name=dataset_info.name, version=args.version, usage=usage, coordinates='absolute')
        longer_edge_size = args.longer_edge_size
        if longer_edge_size:
            longer_edge_size = [int(x.strip()) for x in longer_edge_size.strip().split(',')]
            assert len(longer_edge_size) in [1, 2]

        rotate_angle = args.rotate_angle
        if rotate_angle:
            rotate_angle = [int(x.strip()) for x in rotate_angle.strip().split(',')]
            assert len(rotate_angle) in [1, 2]

        ls_col = []
        ori_ls_col = []
        ra_col = []
        if dataset:
            manifest = dataset.dataset_manifest
            usage_folder = str(usage).split('.')[1]
            base_dir: pathlib.Path = args.output_folder / usage_folder
            base_dir.mkdir(parents=True, exist_ok=True)
            for i, sample in tqdm(enumerate(dataset), desc='Transforming images...'):
                img, _, _ = sample
                format = (args.format and str(args.format).split('.')[1]) or img.format
                file_path = f'{i}.{format}'
                if longer_edge_size:
                    ls = longer_edge_size[0] if len(longer_edge_size) == 1 else random.randint(longer_edge_size[0], longer_edge_size[1])
                    ls_col.append(ls)
                    ori_ls_col.append(max(img.size))
                else:
                    ls = None

                if rotate_angle:
                    ra = rotate_angle[0] if len(rotate_angle) == 1 else random.randint(rotate_angle[0], rotate_angle[1])
                    ra_col.append(ra)
                else:
                    ra = None

                img = process_and_save_image(img, ls, ra, format, base_dir / file_path)
                manifest.images[i].img_path = (pathlib.Path(usage_folder) / file_path).as_posix()
                manifest.images[i].width, manifest.images[i].height = img.size

            coco_dict = coco_generator.run(manifest)
            if args.zip:
                for image in coco_dict['images']:
                    image['zip_file'] = f'{usage_folder}.zip'
            (args.output_folder / f'{usage_folder}.json').write_text(json.dumps(coco_dict, indent=2, ensure_ascii=False), encoding='utf-8')

            if args.zip:
                logger.info(f'Zip folder "{usage_folder}".')
                zip_folder(base_dir, direct=True)

            log_hist('Original Longer Edge Size', ori_ls_col, 5)
            log_hist('Longer Edge Size', ls_col, 5)
            log_hist('Rotate Angle', ra_col, 5)


if __name__ == '__main__':
    main()
