import argparse
import datetime
import json
import pathlib
import os
import zipfile

from vision_datasets import DatasetTypes

TRAIN_USAGE = 'train'
VAL_USAGE = 'val'
TEST_USAGE = 'test'


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            k = os.path.relpath(os.path.join(root, file), path)
            ziph.write(os.path.join(root, file), k)


def create_argparse():
    parser = argparse.ArgumentParser('Prepare the annotation files, data reg json, and zip files for vision-datasets following iris format.')
    parser.add_argument('name', type=str, help="Dataset name.")
    parser.add_argument('--type', '-t', type=str, default=DatasetTypes.IC_MULTICLASS, help="type of dataset.", choices=[DatasetTypes.IC_MULTICLASS, DatasetTypes.IC_MULTILABEL])
    parser.add_argument('--description', '-d', type=str, help="Dataset description.", required=True)
    parser.add_argument('--contact', '-c', type=str, help="contact person.", required=False)
    parser.add_argument('--train_folder', '-tr', type=pathlib.Path, help="Folder including training images.")
    parser.add_argument('--val_folder', '-v', type=pathlib.Path, help="Folder including validation images.")
    parser.add_argument('--test_folder', '-te', type=pathlib.Path, help="Folder including test images.")
    return parser


def main():
    parser = create_argparse()
    args = parser.parse_args()

    labelmap_file = 'labels.txt'
    today = datetime.datetime.now().strftime('%Y%m%d')
    reg_json = {
        'name': args.name,
        'description': args.description,
        'contact': args.contact,
        'version': 1,
        "type": args.type,
        "root_folder": f"classification/{args.name.replace('-', '_')}_{today}",
        "labelmap": labelmap_file,
    }

    folder_by_usage = {
        TRAIN_USAGE: args.train_folder,
        VAL_USAGE: args.val_folder,
        TEST_USAGE: args.test_folder
    }

    classes = os.listdir(folder_by_usage[TRAIN_USAGE])
    reg_json['num_classes'] = len(classes)
    with open(labelmap_file, 'w') as label_out:
        label_out.write('\n'.join(classes))

    n_images = {usage: 0 for usage in folder_by_usage.keys()}

    for usage, folder in folder_by_usage.items():
        if not folder:
            continue

        with open(f'{usage}.txt', 'w') as index_file:
            for i, c in enumerate(classes):
                for img_file in (folder / c).iterdir():
                    img_path = img_file.as_posix()
                    img_path = str(img_path).replace(f'{folder}/', f'{folder}.zip@')
                    index_file.write(f'{img_path} {i}\n')
                    n_images[usage] += 1

        with zipfile.ZipFile(f'{folder}.zip', 'w') as zipf:
            zipdir(folder, zipf)

        reg_json[usage] = {
            "index_path": f"{usage}.txt",
            "files_for_local_usage": [
                f"{folder}.zip"
            ],
            "num_images": n_images[usage]
        }

    pathlib.Path('reg.json').write_text(json.dumps(reg_json, indent=2))


if __name__ == '__main__':
    main()
