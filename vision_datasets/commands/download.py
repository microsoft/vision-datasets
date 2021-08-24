"""Download a dataset from shared storage"""
import argparse
import logging
import os
import shutil

from vision_datasets.common.dataset_downloader import DatasetDownloader
from vision_datasets.common.dataset_registry import DatasetRegistry

logger = logging.getLogger()


def download(dataset_sas_url: str, dataset_name: str, output_dir):
    assert dataset_sas_url
    assert dataset_name
    assert output_dir

    if os.path.exists(output_dir):
        raise RuntimeError(f'Output directory already exists: {output_dir}')

    dataset_registry = DatasetRegistry()
    downloader = DatasetDownloader(dataset_sas_url, dataset_registry)
    with downloader.download(dataset_name) as downloaded:
        for x in downloaded.base_dirs:
            shutil.copytree(x, output_dir)

    logger.info(f'Downloaded {dataset_name} to {output_dir}')


def list_datasets():
    registry = DatasetRegistry()
    for dataset in registry.list_data_version_and_types():
        logger.info(f"Name: {dataset['name']}, version: {dataset['version']}, type: {dataset['type']}")


def convert_to_tsv(dataset, file_path, idx_prefix):
    import json
    from io import BytesIO
    import base64
    from tqdm import tqdm

    idx = 0
    with open(file_path, 'w') as file_out:
        for sample in tqdm(dataset, desc=f'Writing to {file_path}'):
            img_id = f'{idx_prefix}{idx}'
            labels = []
            for label in sample[1]:
                tag_name = dataset.labels[label[0]]
                rect = label[1:5]
                labels.append({'class': tag_name, 'rect': rect})

            buffered = BytesIO()
            sample[0].save(buffered, format='JPEG')
            sample[0].close()

            b64img = base64.b64encode(buffered.getvalue()).decode('utf-8')
            file_out.write(f'{img_id}\t{json.dumps(labels)}\t{b64img}\n')
            idx += 1


def main():
    parser = argparse.ArgumentParser('Download datasets from the shared storage')
    parser.add_argument('dataset_name', nargs='?',
                        help='Dataset name. If not specified, show a list of available datasets')
    parser.add_argument('--output', '-o', help='Output directory.', required=True)
    parser.add_argument('--dataset_sas_url', '-k', help='url to dataset folder.', required=True)
    parser.add_argument('--to_tsv', '-t', help='to tsv format or not.', type=bool, default=False)

    args = parser.parse_args()
    if args.dataset_name:
        if args.to_tsv:
            import tempfile
            import pathlib
            dataset_info = DatasetRegistry().get_dataset_info(args.dataset_name)

            temp_dir = tempfile.mkdtemp()
            try:
                dataset_info.root_folder = pathlib.Path(temp_dir) / pathlib.Path(dataset_info.root_folder)
                download(args.dataset_sas_url, args.dataset_name, dataset_info.root_folder)
                from vision_datasets.common.manifest_dataset import ManifestDataset
                from vision_datasets.common.constants import Usages
                for usage in [Usages.TRAIN_PURPOSE, Usages.TEST_PURPOSE]:
                    dataset = ManifestDataset(dataset_info, usage, coordinates='absolute')
                    if not os.path.exists(args.output):
                        os.mkdir(args.output)
                    convert_to_tsv(dataset, pathlib.Path(args.output) / f'{usage}.tsv', usage)
                    dataset.close()
            finally:
                shutil.rmtree(dataset_info.root_folder)
        else:
            download(args.dataset_sas_url, args.dataset_name, args.output)

    else:
        list_datasets()


if __name__ == '__main__':
    main()
