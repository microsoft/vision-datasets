"""
Download a dataset from shared storage either in original format or converted to TSV
"""

import argparse
import pathlib

from vision_datasets.common import DatasetHub, DatasetRegistry, DatasetTypes
from vision_datasets.commands.utils import add_args_to_locate_dataset_from_name_and_reg_json, convert_to_tsv, get_or_generate_data_reg_json_and_usages, set_up_cmd_logger

logger = set_up_cmd_logger(__name__)

TSV_SUPPORTED_TYPES = [DatasetTypes.IMAGE_CAPTION, DatasetTypes.IMAGE_OBJECT_DETECTION, DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL]


def list_datasets(registry: DatasetRegistry):
    for dataset in registry.list_data_version_and_types():
        logger.info(f"Name: {dataset['name']}, version: {dataset['version']}, type: {dataset['type']}")


def main():
    parser = argparse.ArgumentParser('Download dataset from the shared storage')
    add_args_to_locate_dataset_from_name_and_reg_json(parser)

    parser.add_argument('--to_tsv', '-t', help='to tsv format or not.', action='store_true')

    args = parser.parse_args()
    dataset_reg_json, usages = get_or_generate_data_reg_json_and_usages(args)
    dataset_hub = DatasetHub(dataset_reg_json, args.blob_container, args.local_dir)
    name = args.name
    dataset_info = dataset_hub.dataset_registry.get_dataset_info(name)
    args.local_dir.mkdir(parents=True, exist_ok=True)

    if args.to_tsv:
        if dataset_info.type not in TSV_SUPPORTED_TYPES:
            logger.error(f'Unsupported data type for converting to TSV: {dataset_info.type}.')
            return

        logger.info(f'downloading {name}...')
        for usage in usages:
            dataset_manifest, _, _ = dataset_hub.create_dataset_manifest(name, usage=usage)
            if dataset_manifest is None:
                continue

            logger.info(f'converting {name}, usage {usage} to TSV format...')
            convert_to_tsv(dataset_manifest, pathlib.Path(args.local_dir) / f'{name}-{usage}.tsv')
    else:
        for usage in usages:
            dataset_hub.create_vision_dataset(name, usage=usage)


if __name__ == '__main__':
    main()
