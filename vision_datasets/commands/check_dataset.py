"""
Check if a dataset is prepared well to be consumed by this pkg
"""

import argparse
import pathlib
import random

from tqdm import tqdm

from vision_datasets.common import DatasetHub, DatasetTypes, VisionDataset

from .utils import add_args_to_locate_dataset, get_or_generate_data_reg_json_and_usages, set_up_cmd_logger, is_module_available

logger = set_up_cmd_logger(__name__)


def show_dataset_stats(dataset: VisionDataset):
    logger.info(f'Dataset stats: #images {len(dataset)}')
    if dataset.categories:
        logger.info(f'Dataset stats: #tags {len(dataset.categories)}')


def show_img(sample):
    sample[0].show()
    sample[0].close()

    logger.info(f'annotations = {[str(x) for x in sample[1]]}')


def logging_prefix(dataset_name, version):
    return f'Dataset check {dataset_name}, version {version}: '


def quick_check_images(dataset: VisionDataset):
    show_dataset_stats(dataset)
    for idx in random.sample(range(len(dataset)), min(10, len(dataset))):
        show_img(dataset[idx])


def check_images(dataset: VisionDataset):
    show_dataset_stats(dataset)
    file_not_found_list = []
    for i in tqdm(range(len(dataset)), 'Checking image access..'):
        try:
            dataset[i]
        except (KeyError, FileNotFoundError) as e:
            file_not_found_list.append(str(e))

    if file_not_found_list:
        return ['Files not accessible: ' + (', '.join(file_not_found_list))]

    return []


def _is_integer(bbox):
    return all([isinstance(x, int) or (isinstance(x, float) and x.is_integer()) for x in bbox])


def check_box(bbox, img_w, img_h):
    if len(bbox) != 4 or not _is_integer(bbox):
        return False

    left, t, r, b = bbox
    return left >= 0 and t >= 0 and left < r and t < b and r <= img_w and b <= img_h


def classification_detection_check(dataset: VisionDataset):
    errors = []
    n_imgs_by_class = {x: 0 for x in range(len(dataset.categories))}
    for sample_idx, sample in enumerate(dataset.dataset_manifest.images):
        labels = sample.labels
        c_ids = set([label[0] if dataset.dataset_info.type == DatasetTypes.IMAGE_OBJECT_DETECTION else label for label in labels])
        for c_id in c_ids:
            n_imgs_by_class[c_id] += 1

        if dataset.dataset_info.type == DatasetTypes.IMAGE_OBJECT_DETECTION:
            w, h = sample.width, sample.height
            if not w or not h or w < 0 or h < 0:
                errors.append(f'Image {sample_idx} has invalid width or height: {w}, {h}')
                continue

            for box_id, box in enumerate(labels):
                if not check_box(box[1:], w, h):
                    errors.append(f'Image {sample_idx}, box {box_id} is invalid: {box}\n')

    c_id_with_max_images = max(n_imgs_by_class, key=n_imgs_by_class.get)
    c_id_with_min_images = min(n_imgs_by_class, key=n_imgs_by_class.get)
    mean_images = sum(n_imgs_by_class.values()) / len(n_imgs_by_class)
    stats = {
        'n images': len(dataset),
        'n classes': len(dataset.categories),
        f'max num images per class (cid {c_id_with_max_images})': n_imgs_by_class[c_id_with_max_images],
        f'min num images per class (cid {c_id_with_min_images})': n_imgs_by_class[c_id_with_min_images],
        'mean num images per class': mean_images
    }

    c_ids_with_zero_images = [k for k, v in n_imgs_by_class.items() if v == 0]
    logger.warning(f'Class ids with zero images: {c_ids_with_zero_images}')

    if is_module_available('matplotlib'):
        import matplotlib.pyplot as plt

        plt.hist(list(n_imgs_by_class.values()), density=False, bins=len(set(n_imgs_by_class.values())))
        plt.ylabel('n classes')
        plt.xlabel('n images per class')
        plt.show()

    logger.info(str(stats))

    return errors


def main():
    parser = argparse.ArgumentParser('Check if a dataset is valid for pkg to consume.')
    add_args_to_locate_dataset(parser)
    parser.add_argument('--quick_check', '-q', action='store_true', default=False, help='Randomly check a few data samples from the dataset.')

    args = parser.parse_args()
    prefix = logging_prefix(args.name, args.version)

    data_reg_json, usages = get_or_generate_data_reg_json_and_usages(args)
    dataset_hub = DatasetHub(data_reg_json, args.blob_container, args.local_dir.as_posix())
    dataset_info = dataset_hub.dataset_registry.get_dataset_info(args.name, args.version)

    if not dataset_info:
        logger.error(f'{prefix} dataset does not exist.')
        return

    if args.blob_container and args.local_dir:
        args.local_dir.mkdir(parents=True, exist_ok=True)

    for usage in usages:
        logger.info(f'{prefix} Check dataset with usage: {usage}.')

        # if args.local_dir is none, then this check will directly try to access data from azure blob. Images must be present in uncompressed folder on azure blob.
        dataset = dataset_hub.create_vision_dataset(name=dataset_info.name, version=args.version, usage=usage, coordinates='absolute')
        if dataset:
            err_msg_file = pathlib.Path(f'{args.name}_{usage}_errors.txt')
            errors = []
            if args.data_type in [DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, DatasetTypes.IMAGE_OBJECT_DETECTION]:
                errors.extend(classification_detection_check(dataset))

            if args.quick_check:
                quick_check_images(dataset)
            else:
                errors.extend(check_images(dataset))
            err_msg_file.write_text('\n'.join(errors), encoding='utf-8')
        else:
            logger.info(f'{prefix} No split for {usage} available.')


if __name__ == '__main__':
    main()
