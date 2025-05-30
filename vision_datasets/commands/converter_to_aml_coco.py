import json
import logging
import pathlib
from urllib.parse import urlparse, urlunparse

from tqdm import tqdm

from vision_datasets.commands.utils import add_args_to_locate_dataset, get_or_generate_data_reg_json_and_usages
from vision_datasets.common import FileReader, PILImageLoader, CocoDictGeneratorFactory, DatasetHub, DatasetTypes

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Convert dataset to AML coco format (IC, OD only).')
    add_args_to_locate_dataset(parser)
    parser.add_argument('-o', '--output_dir', required=True, type=pathlib.Path, help='output dir for coco file(s).')

    return parser


def keep_base_url(url_path: str):
    url_parts = urlparse(url_path)
    return urlunparse((url_parts.scheme, url_parts.netloc, url_parts.path, None, None, None))


def main():
    args = create_arg_parser().parse_args()
    assert args.blob_container, '"blob_container" is required for generating "coco_url"'
    assert args.local_dir is None, 'Accessing data from "local_dir" is not supported for now. Data must be present in blob_container.'

    data_reg_json, usages = get_or_generate_data_reg_json_and_usages(args)
    dataset_hub = DatasetHub(data_reg_json, args.blob_container, args.local_dir)
    dataset_info = dataset_hub.dataset_registry.get_dataset_info(args.name, args.version)

    if not dataset_info:
        logger.error(f'dataset {args.name} does not exist.')
        return

    assert dataset_info.type in [DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, DatasetTypes.IMAGE_OBJECT_DETECTION]

    coco_gen = CocoDictGeneratorFactory.create(dataset_info.type)
    file_reader = FileReader()
    for usage in usages:
        manifest, _, _ = dataset_hub.create_dataset_manifest(args.name, version=1, usage=usage)
        if manifest is None:
            logger.info(f'{usage} not exist. Skipping.')
            continue

        coco_dict = coco_gen.run(manifest)
        for image in tqdm(coco_dict['images'], f'{usage}: Processing images...'):
            image['coco_url'] = keep_base_url(image['file_name'])
            if not image.get('width') or not image.get('height'):
                with file_reader.open(image['file_name'], 'rb') as f:
                    img = PILImageLoader.load_from_stream(f)
                    image['width'], image['height'] = img.size
            image['file_name'] = image['coco_url'][len(urlunparse(urlparse(keep_base_url(args.blob_container)))):]

        if dataset_info.type == DatasetTypes.IMAGE_OBJECT_DETECTION:
            image_wh_by_id = {x['id']: (x['width'], x['height']) for x in coco_dict['images']}
            for ann in tqdm(coco_dict['annotations'], f'{usage}: Processing bbox...'):
                w, h = image_wh_by_id[ann['image_id']]
                box = ann['bbox']
                ann['bbox'] = [box[0]/w, box[1]/h, box[2]/w, box[3]/h]

        output_dir = pathlib.Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        coco_filepath = pathlib.Path(args.output_dir) / f'{dataset_info.name}_{usage}.json'
        coco_filepath.write_text(json.dumps(coco_dict, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
