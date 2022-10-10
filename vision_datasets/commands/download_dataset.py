"""
Download a dataset from shared storage either in original format or converted to TSV
"""

import argparse
import os
import tempfile
import pathlib
from vision_datasets.commands.utils import convert_to_tsv, set_up_cmd_logger
from vision_datasets import Usages, DatasetRegistry, DatasetHub, DatasetTypes

logger = set_up_cmd_logger(__name__)

TSV_SUPPORTED_TYPES = [DatasetTypes.IMCAP, DatasetTypes.OD, DatasetTypes.IC_MULTICLASS, DatasetTypes.IC_MULTILABEL]


def list_datasets(registry: DatasetRegistry):
    for dataset in registry.list_data_version_and_types():
        logger.info(f"Name: {dataset['name']}, version: {dataset['version']}, type: {dataset['type']}")


def main():
    parser = argparse.ArgumentParser('Download datasets from the shared storage')
    parser.add_argument('dataset_names', nargs='+', help='Dataset name. If not specified, show a list of available datasets')
    parser.add_argument('--dataset_reg_json', '-r', type=pathlib.Path, required=True)
    parser.add_argument('--output', '-o', help='Output directory.', type=pathlib.Path, required=True)
    parser.add_argument('--dataset_sas_url', '-k', help='url to dataset folder.', required=True)
    parser.add_argument('--to_tsv', '-t', help='to tsv format or not.', action='store_true')

    args = parser.parse_args()
    dataset_names = args.dataset_names
    dataset_hub = DatasetHub(args.dataset_reg_json.read_text())
    if not dataset_names:
        list_datasets(dataset_hub.dataset_registry)
        return

    if not args.output.exists():
        os.makedirs(args.output)

    for dataset_name in dataset_names:
        dataset_info = dataset_hub.dataset_registry.get_dataset_info(dataset_name)
        if args.to_tsv:
            if dataset_info.type not in TSV_SUPPORTED_TYPES:
                logger.warn(f'Unsupported data type for converting to TSV: {dataset_info.type}.')
                continue

            with tempfile.TemporaryDirectory() as temp_dir:
                dataset_info.root_folder = temp_dir / pathlib.Path(dataset_info.root_folder)
                logger.info(f'downloading {dataset_name}...')
                for usage in [Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE]:
                    dataset_manifest = dataset_hub.create_dataset_manifest(args.dataset_sas_url, temp_dir, dataset_name, usage=usage)
                    if not dataset_manifest:
                        continue

                    dataset_manifest = dataset_manifest[0]

                    logger.info(f'converting {dataset_name}, usage {usage} to TSV format...')
                    convert_to_tsv(dataset_manifest, pathlib.Path(args.output) / f'{dataset_name}-{usage}.tsv', '')
        else:
            for usage in [Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE]:
                dataset_hub.create_manifest_dataset(args.dataset_sas_url, args.output, dataset_name, usage=usage)


if __name__ == '__main__':
    main()
