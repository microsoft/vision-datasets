import argparse
import logging
import tempfile

from vision_datasets import DatasetRegistry, Usages, DatasetHub
from vision_datasets.common import ManifestDataset
from vision_datasets.common.util import is_url

logger = logging.getLogger(__name__)


def show_dataset_stats(dataset):
    logger.info(f'Dataset stats: #images {len(dataset)}, #tags {len(dataset.labels)}')


def show_img(img_info):
    if isinstance(img_info[0], str):
        logger.info(f'img sas url {img_info[0]}')
    else:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_info[0]))
        img.show()
        img.close()

    logger.info(f'label = {img_info[1]}, wh = {img_info[2]}')


def logging_prefix(dataset_name):
    return f'Dataset check {dataset_name}: '


def check_dataset(dataset):
    show_dataset_stats(dataset)
    show_img(dataset[0])


def main():
    parser = argparse.ArgumentParser('Check if a dataset is valid')
    parser.add_argument('name', type=str, help="Dataset name.")
    parser.add_argument('--sas_or_dir', '-k', type=str, help="sas url or dataset folder.", required=True)
    parser.add_argument('--reg_json', '-r', type=str, default=None, help="dataset registration json.", required=False)
    parser.add_argument('--local_zip_check_for_sas', '-z', action='store_true', help="check LocalZipDataset support or not when sas is provided.")

    args = parser.parse_args()
    prefix = logging_prefix(args.name)
    dataset_info = DatasetRegistry(args.reg_json).get_dataset_info(args.name)
    if not dataset_info:
        logger.error(f'{prefix} dataset does not exist.')
        return
    else:
        logger.info(f'{prefix} dataset found in registration file.')

    vision_datasets = DatasetHub(args.reg_json)
    is_sas_url = is_url(args.sas_or_dir)
    if is_sas_url:
        for usage in [Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE]:
            logger.info(f'{prefix} Check azure dataset, usage: {usage}')
            dataset = vision_datasets.create_manifest_dataset(args.sas_or_dir, local_dir=None, name=dataset_info.name, usage=usage)
            if dataset:
                show_dataset_stats(dataset)
                show_img(dataset[0])
            else:
                logger.info(f'No split for {usage} available')
    else:
        logger.info(f'{prefix} local dir provided, skipping Azure-based dataset check.')

    if is_sas_url and not args.local_zip_check_for_sas:
        return

    if is_sas_url:
        with tempfile.mkdtemp() as temp_dir:
            for usage in [Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE]:
                logger.info(f'{prefix} Check {ManifestDataset.__name__}, usage: {usage}')
                dataset = vision_datasets.create_manifest_dataset(args.sas_or_dir, local_dir=temp_dir, name=dataset_info.name, usage=usage)
                if dataset:
                    check_dataset(dataset)
                dataset.close()
    else:
        for usage in [Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE]:
            logger.info(f'{prefix} Check {ManifestDataset.__name__}, usage: {usage}')
            dataset = vision_datasets.create_manifest_dataset(container_sas=None, local_dir=args.sas_or_dir, name=dataset_info.name, usage=usage)
            if dataset:
                check_dataset(dataset)
                dataset.close()
            else:
                logger.info(f'No split for {usage} available')


if __name__ == '__main__':
    main()
