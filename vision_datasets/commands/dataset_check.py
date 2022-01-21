import argparse
import logging
import os.path
import pathlib

from vision_datasets import DatasetRegistry, Usages, DatasetHub

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def show_dataset_stats(dataset):
    logger.info(f'Dataset stats: #images {len(dataset)}, #tags {len(dataset.labels)}')


def show_img(sample):

    sample[0].show()
    sample[0].close()

    logger.info(f'label = {sample[1]}, wh = {sample[2]}')


def logging_prefix(dataset_name, version):
    return f'Dataset check {dataset_name}, version {version}: '


def check_dataset(dataset):
    show_dataset_stats(dataset)
    show_img(dataset[0])


def main():
    parser = argparse.ArgumentParser('Check if a dataset is valid')
    parser.add_argument('name', type=str, help="Dataset name.")
    parser.add_argument('--reg_json', '-r', type=str, default=None, help="dataset registration json file path.", required=True)
    parser.add_argument('--version', '-v', type=int, help="Dataset version.", default=None)
    parser.add_argument('--blob_container', '-k', type=str, help="blob container (sas) url", required=False)
    parser.add_argument('--folder_to_check', '-f', type=str, required=False, help="Check the dataset in this folder.")

    args = parser.parse_args()
    prefix = logging_prefix(args.name, args.version)

    data_reg_json = pathlib.Path(args.reg_json).read_text()
    dataset_info = DatasetRegistry(data_reg_json).get_dataset_info(args.name, args.version)
    if not dataset_info:
        logger.error(f'{prefix} dataset does not exist.')
        return
    else:
        logger.info(f'{prefix} dataset found in registration file.')

    vision_datasets = DatasetHub(data_reg_json)

    for usage in [Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE]:
        logger.info(f'{prefix} Check dataset with usage: {usage}.')
        if args.folder_to_check and not os.path.exists(args.folder_to_check):
            os.mkdir(args.folder_to_check)

        # if args.folder_to_check is none, then this check will directly try to access data from azure blob. Images must be present in uncompressed folder on azure blob.
        dataset = vision_datasets.create_manifest_dataset(container_sas=args.blob_container, local_dir=args.folder_to_check, name=dataset_info.name, version=args.version, usage=usage)
        if dataset:
            show_dataset_stats(dataset)
            show_img(dataset[0])
        else:
            logger.info(f'{prefix} No split for {usage} available.')


if __name__ == '__main__':
    main()
