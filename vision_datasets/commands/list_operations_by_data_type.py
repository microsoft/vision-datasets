"""
Download a dataset from shared storage either in original format or converted to TSV
"""

import argparse

from vision_datasets.common import DatasetTypes, SupportedOperationsByDataType
from vision_datasets.commands.utils import set_up_cmd_logger

from .utils import enum_type

logger = set_up_cmd_logger(__name__)


def main():
    parser = argparse.ArgumentParser('List supported operations by data type')

    parser.add_argument('--data_type', '-d', help='list supported operations by data type.', type=enum_type(DatasetTypes), required=True, choices=list(DatasetTypes))
    args = parser.parse_args()
    print(SupportedOperationsByDataType.list(args.data_type))


if __name__ == '__main__':
    main()
