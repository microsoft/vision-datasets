# Iris format

Here is an example with explanation of what a `DatasetInfo` looks like for `iris` format:

```{json}
    {
        "name": "sampled-ms-coco",
        "version": 1,
        "description": "A sampled ms-coco dataset.",
        "type": "object_detection",
        "root_folder": "detection/coco2017_20200401",
        "format": "iris", // indicating the annotation data are stored in iris format
        "train": {
            "index_path": "train_images.txt", // index file for images and labels for training, example can be found in next section
            "files_for_local_usage": [
                "train_images.zip",
                "train_labels.zip"
            ],
        },
        "val": {
            "index_path": "val_images.txt",
            "files_for_local_usage": [
                "val_images.zip",
                "val_labels.zip"
            ],
        },
        "test": {
            "index_path": "test_images.txt",
            "files_for_local_usage": [
                "test_images.zip",
                "test_labels.zip"
            ],
        },
        "labelmap": "labels.txt", // includes tag names
        "image_metadata_path": "image_meta_info.txt", // includes info about image width and height
    },
```

## Iris image classification format

Each rows in the index file (`index_path`) is:

``` {txt}
<image_filepath> <comma-separated-label-indices>
```

Note that the class/label index should start from zero.

Example:

``` {txt}
train_images1.zip@1.jpg 0,1,2
train_images2.zip@1.jpg 2,3
...
```

## Iris object detection format

The index file for OD is slightly different from IC. Each rows in the index file is:

``` {txt}
<image_filepath> <label_filepath>
```

Same with classification, the class/label index should start from 0.

Example for `train_images.txt`:

``` {txt}
train_images.zip@1.jpg train_labels.zip@1.txt
train_images.zip@2.jpg train_labels.zip@2.txt
...
```

Formats and example for a label file like `train_labels.zip@1.txt`:

``` {txt}
class_index left top right bottom
```

``` {txt}
3 200 300 600 1200 // class_id, left, top, right, bottom
4 100 100 200 200
...
```

## Multitask DatasetInfo

The `DatasetInfo` for multitask is not very different from single task. A `'tasks'` section will be found in the json and the `'type'` of the dataset is `'multitask'`. Within each task, it wraps the
info specific to that task.

Below is an example for `'iris'` format, but the general idea applies to `'coco'` format as well.

```{json}
{
    "name": "coco-vehicle-multitask",
    "version": 1,
    "type": "multitask",
    "root_folder": "classification/coco_vehicle_multitask_20210202",
    "format": "iris",
    "tasks": {
        "vehicle_color": {
            "type": "classification_multiclass",
            "train": {
                "index_path": "train_images_VehicleColor.txt",
                "files_for_local_usage": [
                    "train_images.zip"
                ]
            },
            "test": {
                "index_path": "test_images_VehicleColor.txt",
                "files_for_local_usage": [
                    "test_images.zip"
                ]
            },
            "labelmap": "labels_VehicleColor.txt"
        },
        "vehicle_type": {
            "type": "classification_multiclass",
            "train": {
                "index_path": "train_images_VehicleType.txt",
                "files_for_local_usage": [
                    "train_images.zip"
                ]
            },
            "test": {
                "index_path": "test_images_VehicleType.txt",
                "files_for_local_usage": [
                    "test_images.zip"
                ]
            },
            "labelmap": "labels_VehicleType.txt"
        }
    }
}
```
