import argparse
import logging
import os.path
import pathlib
import random
from tqdm import tqdm
from vision_datasets import DatasetRegistry, Usages, DatasetHub, DatasetTypes, ManifestDataset
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def show_dataset_stats(dataset):
    logger.info(f'Dataset stats: #images {len(dataset)}')
    if dataset.labels:
        logger.info(f'Dataset stats: #tags {len(dataset.labels)}')


def show_img(sample):
    sample[0].show()
    sample[0].close()

    logger.info(f'label = {sample[1]}')


def logging_prefix(dataset_name, version):
    return f'Dataset check {dataset_name}, version {version}: '


def quick_check_images(dataset: ManifestDataset):
    show_dataset_stats(dataset)
    for idx in random.sample(range(len(dataset)), min(10, len(dataset))):
        show_img(dataset[idx])


def check_images(dataset: ManifestDataset, err_msg_file: pathlib.Path):
    show_dataset_stats(dataset)
    file_not_found_list = []
    for i in tqdm(range(len(dataset)), 'Checking image access..'):
        try:
            _ = dataset[i]
        except (KeyError, FileNotFoundError) as e:
            file_not_found_list.append(str(e))

    if file_not_found_list:
        err_msg_file.write_text('\n'.join(file_not_found_list))


def classification_detection_check(dataset):
    n_imgs_by_class = {x: 0 for x in range(len(dataset.labels))}
    for sample in dataset.dataset_manifest.images:
        labels = sample.labels
        c_ids = set([label[0] if dataset.dataset_info.type == DatasetTypes.OD else label for label in labels])
        for c_id in c_ids:
            n_imgs_by_class[c_id] += 1

    c_id_with_max_images = max(n_imgs_by_class, key=n_imgs_by_class.get)
    c_id_with_min_images = min(n_imgs_by_class, key=n_imgs_by_class.get)
    mean_images = sum(n_imgs_by_class.values()) / len(n_imgs_by_class)
    stats = {
        'n images': len(dataset),
        'n classes': len(dataset.labels),
        f'max num images per class (cid {c_id_with_max_images})': n_imgs_by_class[c_id_with_max_images],
        f'min num images per class (cid {c_id_with_min_images})': n_imgs_by_class[c_id_with_min_images],
        'mean num images per class': mean_images
    }

    c_ids_with_zero_images = [k for k, v in n_imgs_by_class.items() if v == 0]
    logger.warning(f'Class ids with zero images: {c_ids_with_zero_images}')

    import matplotlib.pyplot as plt

    plt.hist(list(n_imgs_by_class.values()), density=False, bins=len(set(n_imgs_by_class.values())))
    plt.ylabel('n classes')
    plt.xlabel('n images per class')
    plt.show()
    logger.info(str(stats))


def main():
    parser = argparse.ArgumentParser('Check if a dataset is valid')
    parser.add_argument('name', type=str, help='Dataset name.')
    parser.add_argument('--reg_json', '-r', type=pathlib.Path, default=None, help='dataset registration json file path.', required=True)
    parser.add_argument('--version', '-v', type=int, help='Dataset version.', default=None)
    parser.add_argument('--blob_container', '-k', type=str, help='Blob container (sas) url', required=False)
    parser.add_argument('--folder_to_check', '-f', type=pathlib.Path, required=False, help='Check the dataset in this folder.')
    parser.add_argument('--quick_check', '-q', action='store_true', default=False, help='Randomly check a few data samples from the dataset.')

    args = parser.parse_args()
    prefix = logging_prefix(args.name, args.version)

    data_reg_json = args.reg_json.read_text()
    dataset_info = DatasetRegistry(data_reg_json).get_dataset_info(args.name, args.version)
    if not dataset_info:
        logger.error(f'{prefix} dataset does not exist.')
        return
    else:
        logger.info(f'{prefix} dataset found in registration file.')

    vision_datasets = DatasetHub(data_reg_json)

    for usage in [Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE]:
        logger.info(f'{prefix} Check dataset with usage: {usage}.')
        if args.folder_to_check and not args.folder_to_check.exists():
            os.mkdir(args.folder_to_check)

        # if args.folder_to_check is none, then this check will directly try to access data from azure blob. Images must be present in uncompressed folder on azure blob.
        dataset = vision_datasets.create_manifest_dataset(container_sas=args.blob_container, local_dir=args.folder_to_check, name=dataset_info.name, version=args.version, usage=usage)
        if dataset:
            err_msg_file = pathlib.Path(f'{args.name}_{usage}_errors.txt')
            if args.quick_check:
                quick_check_images(dataset)
            else:
                check_images(dataset, err_msg_file)
        else:
            logger.info(f'{prefix} No split for {usage} available.')


if __name__ == '__main__':
    main()
