import argparse
import logging
import os
from PIL import Image

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser('Generate image metainfo needed for object detection datasets.')
    parser.add_argument('index_files', nargs='+', help='index files.')
    parser.add_argument('--output', type=str, help='meta file name', required=True)
    parser.add_argument('--base_folder', type=str, help='base_folder', default='.\\')

    args = parser.parse_args()
    with open(os.path.join(args.base_folder, args.output), 'w') as out_file:
        for file in args.index_files:
            with open(os.path.join(args.base_folder, file), 'r') as file_in:
                for line in file_in:
                    img_id, _ = line.strip().split(' ')
                    img_loc = os.path.join(args.base_folder, img_id.replace('.zip@', '\\'))
                    img = Image.open(img_loc)
                    w, h = img.size
                    out_file.write(f'{img_id} {w} {h}\n')


if __name__ == '__main__':
    main()
