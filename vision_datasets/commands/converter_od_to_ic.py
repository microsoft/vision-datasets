import json
import logging
import multiprocessing
import os
import pathlib
import shutil

from vision_datasets import DatasetHub, Usages
from vision_datasets.common.manifest_dataset import DetectionAsClassificationByCroppingDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Convert OD dataset to ic dataset.')
    parser.add_argument('-n', '--name', type=str, required=True, help='dataset name')
    parser.add_argument('-r', '--reg_json_path', type=str, default=None, help="dataset registration json path.", required=True)
    parser.add_argument('-k', '--sas', type=str, help="sas url.", required=False, default=None)
    parser.add_argument('-l', '--local_folder', type=str, help="detection dataset folder.", required=False)
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='target folder of the converted classification dataset')
    parser.add_argument('-zb', '--zoom_ratio_bounds', type=str, required=False,
                        help='lower and bound of the ratio that box height and width can expand (>1) or shrink (0-1), during cropping, e.g, 0.8/1.2')
    parser.add_argument('-sb', '--shift_relative_bounds', type=str, required=False,
                        help='lower/upper bounds of relative ratio wrt box width and height that a box can shift, during cropping, e.g., "-0.3/0.1"')
    parser.add_argument('-np', '--n_copies', type=int, required=False, default=1, help='number of copies per bbox')
    parser.add_argument('-s', '--rnd_seed', type=int, required=False, help='random see for box expansion/shrink/shifting.', default=0)
    parser.add_argument('--zip', dest='zip', action='store_true', help='Flag to add zip prefix to the image paths.')

    return parser


def process_phase(params):
    args, aug_params, phase = params

    logger.info(f'download dataset manifest for {args.name}...')
    dataset_resources = DatasetHub(pathlib.Path(args.reg_json_path).read_text())
    dataset = dataset_resources.create_manifest_dataset(args.sas, args.local_folder, args.name, usage=phase, coordinates='absolute')
    if not dataset:
        logger.info(f'Skipping non-existent phase {phase}.')
        return

    logger.info(f'start conversion for {args.name}...')
    ic_dataset = DetectionAsClassificationByCroppingDataset(dataset, aug_params)
    manifest = ic_dataset.generate_manifest(dir=phase, n_copies=args.n_copies)
    if args.zip:
        for img in manifest.images:
            img.img_path = f'{phase}.zip@{img.img_path}'

    coco = manifest.generate_coco_annotations()
    with open(f'{args.output_folder}/{phase}.json', 'w') as coco_out:
        coco_out.write(json.dumps(coco, indent=2))
    shutil.move(f'{phase}', f'{args.output_folder}/', copy_function=shutil.copytree)


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    aug_params = {}
    if args.zoom_ratio_bounds:
        low, up = args.zoom_ratio_bounds.split('/')
        aug_params['zoom_ratio_bounds'] = (float(low), float(up))

    if args.shift_relative_bounds:
        low, up = args.shift_relative_bounds.split('/')
        aug_params['shift_relative_bounds'] = (float(low), float(up))

    if aug_params:
        aug_params['rnd_seed'] = args.rnd_seed

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if args.local_folder and not os.path.exists(args.local_folder):
        os.makedirs(args.local_folder)

    params = [(args, aug_params, phase) for phase in [Usages.TRAIN_PURPOSE, Usages.VAL_PURPOSE, Usages.TEST_PURPOSE]]
    with multiprocessing.Pool(3) as pool:
        pool.map(process_phase, params)


if __name__ == '__main__':
    main()
