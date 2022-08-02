""" 
target coco format expamle: https://irisdataset.blob.core.windows.net/uvsdatasets/b92_regular_od/train_images.json
example inputs 

irisCoco_path = r'mobileone_shelf_test_sample_annotations_new_categories_fix_labelmap.json'
out_put_file = "mobile_one_AML_coco_irisdataset_test.json"

img_path = 'images/shelf_images'
absolute_url = 'https://irisdataset.blob.core.windows.net/objectdetection/mobileone_jiahe/images/shelf_images'
coco_url = 'AmlDatastore://mobileone/'
date_captured = '2022-06-24T23:34:17.7980622Z'

has_bbox = True
"""
import json
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Convert IRIS coco to AML coco format.')
    parser.add_argument('-inputCocoName', '--irisCoco_path', required=True, type=str, help='Iris coco file to convert.')
    parser.add_argument('-outputJsonName', '--out_put_file', required=True, type=str, default='AML_coco.json', help='output json name.')
    parser.add_argument('-imagePath', '--img_path', required=True, type=str, default='images/shelf_images', help='input image path.')
    parser.add_argument('-absoluteUrl', '--absolute_url', required=True, type=str, default='https://irisdataset.blob.core.windows.net/objectdetection/mobileone_jiahe/images/shelf_images', help='absolute url for blob container.')
    parser.add_argument('-hasBbox', '--has_bbox', required=True, type=bool, default=True, help='input image path.')

    parser.add_argument('-projectUrl', '--coco_url', required=False, type=str, default='AmlDatastore://mobileone/', help='project url.')
    parser.add_argument('-dateCaptured', '--date_captured', required=False, type=str, default='2022-06-24T23:34:17.7980622Z', help='date captured.')

    return parser

def main():
    args = create_arg_parser().parse_args()
    logger.info(args.__dict__)

    standard_images = []
    dir_id_WH = {}
    standard_annotations = []

    with open(args.irisCoco_path) as f:
        iris_file = json.load(f)

        iris_images = iris_file['images']
        iris_annotations = iris_file["annotations"]
        iris_categories = iris_file['categories']

        for image in iris_images:
            standard_tmp = {}
            WH = []
            standard_tmp['id'] = image['id']
            standard_tmp['width'] = image['width']
            standard_tmp['height'] = image['height']
            standard_tmp['file_name'] = args.img_path + '/' + image['file_name'].split('/')[-1]
            standard_tmp['coco_url'] = 'AmlDatastore://mobileone/' + image['file_name'].split('/')[-1]
            standard_tmp['absolute_url'] = args.absolute_url + '/' + image['file_name'].split('/')[-1]
            standard_tmp['date_captured'] = args.date_captured
            WH.append(image['width'])
            WH.append(image['height'])
            dir_id_WH[image['id']] = WH

            standard_images.append(standard_tmp)

        for annotation in iris_annotations:
            standard_tmp = {}
            standard_tmp['id'] = annotation['id']
            standard_tmp['category_id'] = annotation['category_id']
            standard_tmp['image_id'] = annotation['image_id']
            standard_tmp['area'] = annotation['area']
            
            if args.has_bbox:
                bbox = []
                bbox.append(annotation['bbox'][0]/dir_id_WH[annotation['image_id']][0])
                bbox.append(annotation['bbox'][1]/dir_id_WH[annotation['image_id']][1])
                bbox.append(annotation['bbox'][2]/dir_id_WH[annotation['image_id']][0])
                bbox.append(annotation['bbox'][3]/dir_id_WH[annotation['image_id']][1])
                standard_tmp['bbox'] = bbox

            standard_annotations.append(standard_tmp) 

    standard_coco = {}
    standard_coco['images'] = standard_images
    standard_coco['annotations'] = standard_annotations
    standard_coco['categories'] = iris_categories

    out_file = open(args.out_put_file, "w")
    json.dump(standard_coco, out_file, indent=4)
    out_file.close()
    
if __name__ == '__main__':
    main()