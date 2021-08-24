import os
import pathlib

from vision_datasets import DatasetHub, Usages


def create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Convert OD dataset to ic dataset.')
    parser.add_argument('-t', '--target_folder', type=str, required=True, help='target folder of the converted dataset')
    parser.add_argument('-k', '--sas_or_dir', type=str, help="sas url or dataset folder.", required=True)
    parser.add_argument('-r', '--reg_json', type=str, default=None, help="dataset registration json.", required=False)
    parser.add_argument('-n', '--name', type=str, required=True, help='dataset name')

    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    dataset_resources = DatasetHub(pathlib.Path(args.json_path).read_text())

    if not os.path.exists(args.target_folder):
        os.makedirs(args.target_folder)

    for phase in [Usages.TRAIN_PURPOSE, Usages.TEST_PURPOSE]:
        img_folder = os.path.join(args.target_folder, phase)
        img_index_file = os.path.join(args.target_folder, f'{phase}.txt')
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)
        print('download dataset manifest...')
        az_dataset = dataset_resources.create_manifest_dataset(args.sas, None, args.name, usage=phase, coordinates='absolute')
        print('start conversion...')
        with open(img_index_file, 'w') as index_file_out:
            for img, labels, idx in az_dataset:
                print(f'image idx {idx} {img.size}')
                crop_idx = 0
                for c_idx, l, t, r, b in labels:
                    print((l, t, r, b))
                    crop_img = img.crop((l, t, r, b))
                    crop_id = f'{idx}-{crop_idx}'
                    crop_idx += 1
                    crop_img.save(os.path.join(img_folder, f'{crop_id}.jpg'), "JPEG")
                    index_file_out.write(f'{phase}.zip@{crop_id}.jpg {c_idx}\n')


if __name__ == '__main__':
    main()
