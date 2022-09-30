"""Download a dataset from shared storage"""
import argparse
import logging
import os
import shutil
import tempfile
import pathlib
from vision_datasets.common.constants import DatasetTypes

from vision_datasets.common.dataset_downloader import DatasetDownloader
from vision_datasets import Usages, DatasetRegistry, DatasetHub
from vision_datasets.common.manifest_dataset import ManifestDataset

logger = logging.getLogger()
TSV_SUPPORTED_TYPES = [DatasetTypes.IMCAP, DatasetTypes.OD, DatasetTypes.IC_MULTICLASS, DatasetTypes.IC_MULTILABEL]


def download(dataset_registry: DatasetRegistry, dataset_sas_url: str, dataset_name: str, dest: pathlib.Path):
    assert dataset_registry
    assert dataset_sas_url
    assert dataset_name
    assert dest

    if dest.exists():
        shutil.rmtree(dest)

    downloader = DatasetDownloader(dataset_sas_url, dataset_registry)
    with downloader.download(dataset_name) as downloaded:
        for x in downloaded.base_dirs:
            shutil.copytree(x, dest)


def list_datasets(registry: DatasetRegistry):
    for dataset in registry.list_data_version_and_types():
        logger.info(f"Name: {dataset['name']}, version: {dataset['version']}, type: {dataset['type']}")


def convert_to_tsv(dataset: ManifestDataset, file_path, idx_prefix):
    import json
    from io import BytesIO
    import base64
    from tqdm import tqdm

    idx = 0
    with open(file_path, 'w', encoding='utf-8') as file_out:
        for img, labels, _ in tqdm(dataset, desc=f'Writing to {file_path}'):
            img_id = f'{idx_prefix}{idx}'
            converted_labels = []
            for label in labels:
                if dataset.dataset_manifest.data_type in [DatasetTypes.IC_MULTILABEL, DatasetTypes.IC_MULTICLASS]:
                    tag_name = dataset.labels[label]
                    converted_label = {'class': tag_name}
                elif dataset.dataset_manifest.data_type == DatasetTypes.OD:
                    tag_name = dataset.labels[label[0]]
                    rect = label[1:5]
                    converted_label = {'class': tag_name, 'rect': rect}
                elif dataset.dataset_manifest.data_type == DatasetTypes.IMCAP:
                    converted_label = {'caption': label}

                converted_labels.append(converted_label)

            buffered = BytesIO()
            img.save(buffered, format='JPEG')
            img.close()

            b64img = base64.b64encode(buffered.getvalue()).decode('utf-8')
            file_out.write(f'{img_id}\t{json.dumps(converted_labels, ensure_ascii=False)}\t{b64img}\n')
            idx += 1


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
                    dataset = dataset_hub.create_manifest_dataset(args.dataset_sas_url, temp_dir, dataset_name, usage=usage)
                    if not dataset:
                        continue

                    logger.info(f'converting {dataset_name}, usage {usage} to TSV format...')
                    convert_to_tsv(dataset, pathlib.Path(args.output) / f'{dataset_name}-{usage}.tsv', '')
                    dataset.close()
        else:
            for usage in [Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE]:
                dataset_hub.create_manifest_dataset(args.dataset_sas_url, args.output, dataset_name, usage=usage)


if __name__ == '__main__':
    main()
