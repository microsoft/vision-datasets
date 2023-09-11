"""
This script converts a dataset from vision_datasets or COCO JSON into TSV or JSONL format.
Each line is an image-oriented representation of an image and its annotations.
"""

import argparse
import pathlib
from enum import Enum

from vision_datasets.common import DatasetHub
from vision_datasets.commands.utils import add_args_to_locate_dataset, convert_to_tsv, convert_to_jsonl, enum_type, get_or_generate_data_reg_json_and_usages, set_up_cmd_logger

logger = set_up_cmd_logger(__name__)


class LineFormat(Enum):
    JSONL = 'jsonl'
    TSV = 'tsv'


def logging_prefix(dataset_name, version, format):
    return f'Dataset {dataset_name} version {version}, convert to {format}:'


def main():
    parser = argparse.ArgumentParser('Convert a dataset to TSV(s) or JONSL(s).')
    add_args_to_locate_dataset(parser)
    parser.add_argument('--format', '-fm', type=enum_type(LineFormat), default=LineFormat.JSONL, help='Format of output data.', choices=list(LineFormat), required=False)
    parser.add_argument('--flatten', '-fl', action='store_true', help="If an image has multiple annotations, one image will be flattend in to multiple entries with image being duplicated.")
    parser.add_argument('--output_dir', '-o', type=pathlib.Path, required=False, default=pathlib.Path('./'), help='File(s) will be saved here.')

    args = parser.parse_args()
    prefix = logging_prefix(args.name, args.version, args.format)

    data_reg_json, usages = get_or_generate_data_reg_json_and_usages(args)

    hub = DatasetHub(data_reg_json, args.blob_container, args.local_dir.as_posix())
    if not hub.dataset_registry.get_dataset_info(args.name, args.version):
        raise RuntimeError(f'{prefix} dataset does not exist.')

    if args.blob_container and args.local_dir:
        args.local_dir.mkdir(parents=True, exist_ok=True)

    for usage in usages:
        logger.info(f'{prefix} Check dataset with usage: {usage}.')
        # if args.local_dir is none, then this check will directly try to access data from azure blob. Images must be present in uncompressed folder on azure blob.
        manifest, _, _ = hub.create_dataset_manifest(name=args.name, version=args.version, usage=usage)
        if manifest is None:
            logger.info(f'{prefix} No split for {usage} available.')
        else:
            if args.format == LineFormat.JSONL:
                convert_to_jsonl(manifest, args.output_dir / f"{args.name}.{usage}.jsonl", args.flatten)
            else:
                convert_to_tsv(manifest, args.output_dir / f"{args.name}.{usage}.tsv")


if __name__ == '__main__':
    main()
