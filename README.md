# Vision Datasets

## Introduction

This repo

- defines the contract for dataset for purposes such as training, visualization, and exploration
- provides API for organizing and accessing datasets: `DatasetHub`

## Dataset Contracts

- `DatasetManifest` wraps the information about a dataset including labelmap, images (width, height, path to image), and annotations. `ImageDataManifest` encapsulates information about each image.
- `ImageDataManifest` encapsulates image-specific information, such as image id, path, labels, and width/height. One thing to note here is that the image path can be
    1. a local path (absolute `c:\images\1.jpg` or relative `images\1.jpg`)
    2. a local path in a **non-compressed** zip file (absolute `c:\images.zip@1.jpg` or relative `images.zip@1.jpg`) or
    3. an url
- `ManifestDataset` is an iterable dataset class that consumes the information from `DatasetManifest`.

`ManifestDataset` is able to load the data from all three kinds of paths. Both 1. and 2. are good for training, as they access data from local disk while the 3rd one is good for data exploration, if you have the data in azure storage.

Currently, three basic types of data are supported: `classification_multilabel`, `classification_multiclass`, and `object_detection`. `multitask` type is a composition type, where one set of images has multiple sets of annotations available for different tasks, where each task can be of the three basic types.

For `multitask` dataset, the labels stored in the `ImageDataManifest` is a `dict` mapping from task name to that task's labels. The labelmap stored in `DatasetManifest` is also a `dict` mapping from task name to that task's labels.

### Creating DatasetManifest

In addition to loading a serialized `DatasetManifest` for instantiation, this repo currently supports two formats of data that can instantiates `DatasetManifest`,
using `DatasetManifest.create_dataset_manifest(dataset_info, usage, container_sas_or_root_dir)`: `IRIS` and `COCO`.

`DatasetInfo` as the first arg in the arg list wraps the metainfo about the dataset like the name of the dataset, locations of the images, annotation files, etc. See examples in the sections below
for different data formats.

Once a `DatasetManifest` is created, you can create a `ManifestDataset` for accessing the dataset:

```{python}
dataset = ManifestDataset(dataset_info, dataset_manifest, coordinates='relative')
```

#### Coco format

Here is an example with explanation of what a `DatasetInfo` looks like for coco format, when it is serialized into json:

```{json}
    {
        "name": "sampled-ms-coco",
        "version": 1,
        "description": "A sampled ms-coco dataset.",
        "type": "object_detection",
        "format": "coco", // indicating the annotation data are stored in coco format
        "root_folder": "detection/coco2017_20200401", // a root folder for all files listed
        "train": {
            "index_path": "train.json", // coco json file for training, see next section for example
            "files_for_local_usage": [ // associated files including data such as images
                "train_images.zip"
            ]
        },
        "val": {
            "index_path": "val.json",
            "files_for_local_usage": [
                "test_images.zip"
            ]
        },
        "test": {
            "index_path": "test.json",
            "files_for_local_usage": [
                "test_images.zip"
            ]
        }
    }
```

##### Coco JSON - Image classification

Here is one example of the train.json, val.json, or test.json in the `DatasetInfo` above. Note that the `"id"` for `images`, `annotations` and `categories` should be consecutive integers, **starting from 1**. Note that our lib might work with id starting from 0, but many tools like [CVAT](https://github.com/openvinotoolkit/cvat/issues/2085) and official [COCOAPI](https://github.com/cocodataset/cocoapi/issues/507) will fail.

``` {json}
{
  "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "train_images.zip@siberian-kitten.jpg"},
              {"id": 2, "width": 224.0, "height": 224.0, "file_name": "train_images.zip@kitten 3.jpg"}],
              //  file_name is the image path, which supports three formats as described in previous section.
  "annotations": [
      {"id": 1, "category_id": 1, "image_id": 1},
      {"id": 2, "category_id": 1, "image_id": 2},
      {"id": 3, "category_id": 2, "image_id": 2}
  ],
  "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
}
```

##### Coco JSON - Object detection

``` {json}
{
  "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "train_images.zip@siberian-kitten.jpg"},
              {"id": 2, "width": 224.0, "height": 224.0, "file_name": "train_images.zip@kitten 3.jpg"}],
  "annotations": [
      {"id": 1, "category_id": 1, "image_id": 1, "bbox": [10, 10, 100, 100]},
      {"id": 2, "category_id": 1, "image_id": 2, "bbox": [100, 100, 200, 200]},
      {"id": 3, "category_id": 2, "image_id": 2, "bbox": [20, 20, 200, 200]}
  ],
  "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
}
```

bbox format should be **absolute** pixel position following either `ltwh: [left, top, width, height]` or `ltrb: [left, top, right, bottom]`. `ltwh` is the default format. To work with `ltrb`, please specify `bbox_format` to be `ltrb` in coco json file.

Note that

- Note that we used to use `ltrb` as default. If your coco annotations were prepared to work with this repo before version 0.1.2. Please add `"bbox_format": "ltrb"` to your coco file.
- Regardless of what format bboxes are stored in Coco file, when annotations are transformed into `ImageDataManifest`, the bbox will be unified into `ltrb: [left, top, right, bottom]`.

#### Iris format

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

##### Iris image classification format

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

##### Iris object detection format

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

#### Multitask DatasetInfo

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

## Dataset management and access

Once you have multiple datasets, it is more convenient to have all the `DatasetInfo` in one place and instantiate `DatasetManifest` or even `ManifestDataset` by just using the dataset name, usage (
train, val ,test) and version.

This repo offers the class `DatasetHub` for this purpose. Once instantiated with a json including the `DatasetInfo` for all datasets, you can retrieve a `ManifestDataset` by

```{python}
import pathlib

dataset_infos_json_path = 'datasets.json'
dataset_hub = DatasetHub(pathlib.Path(dataset_infos_json_path).read_text())
stanford_cars = dataset_hub.create_manifest_dataset(blob_container_sas, local_dir, 'stanford-cars', version=1, usage='train')

for img, targets, sample_idx_str in stanford_cars:
    img.show()
    img.close()
    print(targets)
```

Note that this hub class works with data saved in both Azure Blob container and on local disk.

If `local_dir`:

1. is provided, the hub will look for the resources locally and **download the data** (files included in "
   files_for_local_usage", the index files, metadata (if iris format), labelmap (if iris format))
   from `blob_container_sas` if not present locally
2. is NOT provided (i.e. `None`), the hub will create a manifest dataset that directly consumes data from the blob
   indicated by `blob_container_sas`. Note that this does not work, if data are stored in zipped files. You will have to
   unzip your data in the azure blob. (Index files requires no update, if image paths are for zip files: "a.zip@1.jpg").
   This kind of azure-based dataset is good for large dataset exploration, but can be slow for training.

When data exists on local disk, `blob_container_sas` can be `None`.

### Training with PyTorch

Training with PyTorch is easy. After instantiating a `ManifestDataset`, simply passing it in `vision_datasets.pytorch.torch_dataset.TorchDataset` together with the `transform`, then you are good to go with the PyTorch DataLoader for training.

### Managing datasets with DatasetHub on cloud storage

If you are using `DatasetHub` to manage datasets in cloud storage, we recommend zipping (with uncompressed mode) the images into one or multiple zip files before uploading it and update the file path in index files to be like `train.zip@1.jpg` from `train\1.jpg`. You can do it with `7zip` (set compression level to 'store') on Windows or [zip](https://superuser.com/questions/411394/zip-files-without-compression) command on Linux.

If you upload folders of images directly to cloud storage:

- you will have to list all images in `"files_for_local_usage"`, which can be millions of entries
- downloading images one by one (even with multithreading) is much slower than downloading a few zip files

One more thing is that sometimes when you create a zip file `train.zip`, you might find out that there is only one `train` folder in the zip. This will fail the file loading if the path is `train.zip@1.jpg`, as the image is actually at `train.zip@train\1.jpg`. It is usually a good idea to avoid this extra layer of folder when zipping and double-confirm this does not happen by mistake.
