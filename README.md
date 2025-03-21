# Vision Datasets

## Introduction

This repo

- defines unified contract for dataset for purposes such as training, visualization, and exploration, via `DatasetManifest`, `ImageDataManifest`, etc.
- provides many commonly used dataset operation, such as sample dataset by categories, sample few-shot sub-dataset, sample dataset by ratios, train-test split, merge dataset, etc. (See [Here](#oom))
- provides API for organizing and accessing datasets, via `DatasetHub`

Currently, seven `basic` types of data are supported:

- `image_classification_multiclass`: each image can is only with one label.
- `image_classification_multilabel`: each image can is with one or multiple labels (e.g., 'cat', 'animal', 'pet').
- `image_object_detection`: each image is labeled with bounding boxes surrounding the objects of interest.
- `image_text_matching`: each image is associated with a collection of texts describing the image, and whether each text description matches the image or not.
- `image_matting`: each image has a pixel-wise annotation, where each pixel is labeled as 'foreground' or 'background'.
- `image_regression`: each image is labeled with a real-valued numeric regression target.
- `image_caption`: each image is labeled with a few texts describing the images.
- `text_2_image_retrieval`: each image is labeled with a number of text queries describing the image. Optionally, an image is associated with one label.
- `visual_question_answering`: each image is labeled with a number of question-answer pairs
- `visual_object_grounding`: each image is labeled with a number of question-answer-bboxes triplets.

`multitask` type is a composition type, where one set of images has multiple sets of annotations available for different tasks, where each task can be of any basic type.

`key_value_pair` type is a generalized type, where a sample can be one or multiple images with optional text, labeled with key-value pairs. The keys and values are defined by a schema. Note that all the above seven basic types can be defined as this type with specific schemas.

**Note that `image_caption` and `text_2_image_retrieval` might be merged into `image_text_matching` in future.**

## Dataset Contracts

We support datasets with two types of annotations:

- single-image annotation (S), and
- multi-image annotation (M)

Below table shows all the supported contracts: 
| Annotation | Contract class                       | Explaination                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| :--------- | :----------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S          | `DatasetManifest`                    | wraps the information about a dataset including labelmap, images (width, height, path to image), and annotations. Information about each image is obtained in `ImageDataManifest`. <br>For multitask dataset, the labels stored in the ImageDataManifest is a dict mapping from task name to that task's labels. The labelmap stored in DatasetManifest is also a dict mapping from task name to that task's labels.                            |
| S,M        | `ImageDataManifest`                  | encapsulates image-specific information, such as image id, path, labels, and width/height. One thing to note here is that the image path can be:<br>&nbsp;1. a local path (absolute `c:\images\1.jpg` or relative `images\1.jpg`), <br>&nbsp;2. a local path in a **non-compressed** zip file (absolute `c:\images.zip@1.jpg` or relative `images.zip@1.jpg`) or <br>&nbsp;3. an url. <br>All three kinds of paths can be loaded by `VisionDataset` |
| S          | `ImageLabelManifest`                 | encapsulates one single image-level annotation                                                                                                                                                                                                                                                                                                                                                                                                      |
| S          | `CategoryManifest`                   | encapsulates the information about a category, such as its name and super category, if applicable                                                                                                                                                                                                                                                                                                                                                   |
| M          | `MultiImageLabelManifest`            | is abstract class. It encapsulates one annotation with one or multiple images, each image is stored as an image index.                                                                                                                                                                                                                                                                                                                              |
| M          | `DatasetManifestWithMultiImageLabel` | supports annotations associated with one or multiple images. Each annotation is represented by `MultiImageLabelManifest` class, and each image is represented by `ImageDataManifest`.                                                                                                                                                                                                                                                               |
| M          | `KeyValuePairDatasetManifest`        | inherits `DatasetManifestWithMultiImageLabel`, dataset with each sample having `KeyValuePairLabelManifest` label, dataset is also associated with a schema to define the expected keys and values.                                                                                                                                                                                                                                                  |
| M          | `KeyValuePairLabelManifest`          | inherits `MultiImageLabelManifest`, encapsulates label information of `KeyValuePairDatasetManifest`. Each label has fields `img_ids` (associated images), `text` (associated text input), and `fields` (dictionary of interested field keys and values).                                                                                                                                                                                   |
| S,M        | `VisionDataset`                      | is an iterable dataset class that consumes the information from `DatasetManifest` or `DatasetManifestWithMultiImageLabel`                                                                                                                                                                                                                                                                                                                           |

### Creating DatasetManifest

In addition to loading a serialized `DatasetManifest` for instantiation, this repo currently supports two formats of data that can instantiates `DatasetManifest`,
using `DatasetManifest.create_dataset_manifest(dataset_info, usage, container_sas_or_root_dir)`: `COCO` and `IRIS` (legacy).

`DatasetInfo` as the first arg in the arg list wraps the metainfo about the dataset like the name of the dataset, locations of the images, annotation files, etc. See examples in the sections below
for different data formats.

Once a `DatasetManifest` is created, you can create a `VisionDataset` for accessing the data in the dataset, especially the image data, for training, visualization, etc:

```{python}
dataset = VisionDataset(dataset_info, dataset_manifest, coordinates='relative')
```


### Creating KeyValuePairDatasetManifest

You can use `CocoManifestAdaptorFactory` to create the manifest from COCO format data and a schema, a COCO data example can be found in `COCO_DATA_FORMAT.md`, and a schema example (dictionary) can be found in `DATA_PREPARATION.md`. 

```{python}
from vision_datasets.common import CocoManifestAdaptorFactory, DatasetTypes
# check schema dictionary example From `DATA_PREPARATION.md`
adaptor = CocoManifestAdaptorFactory.create(DatasetTypes.KEY_VALUE_PAIR, schema=schema_dict)
key_value_pair_dataset_manifest = adaptor.create_dataset_manifest(coco_file_path_or_url='test.json', url_or_root_dir='data/')  # image paths in test.json is relative to url_or_root_dir
# test the first sample
print(
    key_value_pair_dataset_manifest.images[0].img_path,'\n',
    key_value_pair_dataset_manifest.annotations[0].fields,'\n',
    key_value_pair_dataset_manifest.annotations[0].text,'\n',
)
```

Once a `KeyValuePairDatasetManifest` is created, along with a dataset_info, create a `VisionDataset` for accessing the data in the dataset.

```{python}
from vision_datasets.common import DatasetInfoFactory, VisionDataset
# check dataset information dictionary example From `DATA_PREPARATION.md`
dataset_info = DatasetInfoFactory.create(dataset_info_dict)
dataset = VisionDataset(dataset_info, key_value_pair_dataset_manifest)
# test the first sample
imgs, target, _ = dataset[0]
print(imgs)
print(target)
```

### Loading IC/OD/VQA Datasets in KeyValuePair (KVP) Format:
You can convert an existing IC/OD VisionDataset to the generalized KVP format using the following adapter:

```{python}
# For MultiClass and MultiLabel IC dataset
from vision_datasets.image_classification import MulticlassClassificationAsKeyValuePairDataset, MultilabelClassificationAsKeyValuePairDataset
sample_multiclass_ic_dataset = VisionDataset(dataset_info, dataset_manifest)
kvp_dataset = MulticlassClassificationAsKeyValuePairDataset(sample_multiclass_ic_dataset)
sample_multilabel_ic_dataset = VisionDataset(dataset_info, dataset_manifest)
kvp_dataset = MultilabelClassificationAsKeyValuePairDataset(sample_multilabel_ic_dataset)


# For OD dataset
from vision_datasets.image_object_detection import DetectionAsKeyValuePairDataset, DetectionAsKeyValuePairDatasetForMultilabelClassification
sample_od_dataset = VisionDataset(dataset_info, dataset_manifest)
kvp_dataset = DetectionAsKeyValuePairDataset(sample_od_dataset)
kvp_dataset_for_multilabel_classification = DetectionAsKeyValuePairDatasetForMultilabelClassification(sample_od_dataset)

# For VQA dataset
from vision_datasets.visual_question_answering import VQAAsKeyValuePairDataset
sample_vqa_dataset = VisionDataset(dataset_info, dataset_manifest)
kvp_dataset = VQAAsKeyValuePairDataset(sample_vqa_dataset)
```


#### Coco format

Here is an example with explanation of what a `DatasetInfo` looks like for coco format, when it is serialized into json:

```json
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
                "images/train_images.zip"
            ]
        },
        "val": {
            "index_path": "val.json",
            "files_for_local_usage": [
                "images/val_images.zip"
            ]
        },
        "test": {
            "index_path": "test.json",
            "files_for_local_usage": [
                "images/test_images.zip"
            ]
        }
    }
```

Coco annotation format details w.r.t. `image_classification_multiclass/label`, `image_object_detection`, `image_caption`, `image_text_match`, `key_value_pair`, and `multitask`  can be found in `COCO_DATA_FORMAT.md`.

Index file can be put into a zip file as well (e.g., `annotations.zip@train.json`), no need to add the this zip to "files_for_local_usage" explicitly.

#### Iris format

Iris format is a legacy format which can be found in `IRIS_DATA_FORMAT.md`. Only `multiclass/label_classification`, `object_detection` and `multitask` are supported.

## Dataset management and access

Check [DATA_PREPARATION.md](DATA_PREPARATION.md) for complete guide on how to prepare datasets in steps.

Once you have multiple datasets, it is more convenient to have all the `DatasetInfo` in one place and instantiate `DatasetManifest` or even `VisionDataset` by just using the dataset name, usage (
train, val ,test) and version.

This repo offers the class `DatasetHub` for this purpose. Once instantiated with a json including the `DatasetInfo` for all datasets, you can retrieve a `VisionDataset` by

```python
import pathlib
from vision_datasets.common import Usages, DatasetHub

dataset_infos_json_path = 'datasets.json'
dataset_hub = DatasetHub(pathlib.Path(dataset_infos_json_path).read_text(), blob_container_sas, local_dir)
stanford_cars = dataset_hub.create_vision_dataset('stanford-cars', version=1, usage=Usages.TRAIN)

# note that you can pass multiple datasets.json to DatasetHub, it can combine them all
# example: DatasetHub([ds_json1, ds_json2, ...])
# note that you can specify multiple usages in create_vision_dataset call
# example dataset_hub.create_vision_dataset('stanford-cars', version=1, usage=[Usages.TRAIN, Usages.VAL])

for img, targets, sample_idx_str in stanford_cars:
    if isinstance(img, list):  # for key_value_pair dataset, the first item is a list of images
       img = img[0]
    img.show()
    img.close()
    print(targets)
    input()
```

Note that this hub class works with data saved in both Azure Blob container and on local disk.

If `local_dir`:

1. is provided, the hub will look for the resources locally and **download the data** (files included in "
   files_for_local_usage", the index files, metadata (if iris format), labelmap (if iris format))
   from `blob_container_sas` if not present locally
2. is NOT provided (i.e. `None`), the hub will create a manifest dataset that directly consumes data from the blob
   indicated by `blob_container_sas`. Note that this does not work, if data are stored in zipped files. You will have to
   unzip your data in the azure blob. (Index files requires no update, if image paths are for zip files: `a.zip@1.jpg`).
   This kind of azure-based dataset is good for large dataset exploration, but can be slow for training.

When data exists on local disk, `blob_container_sas` can be `None`.

## Operations on manifests {#oom}

There are supported operations on manifests for different data types, such as split, merge, sample, etc. You can run

`vision_list_supported_operations -d {DATA_TYPE}`

to see the supported operations for a specific data type. You can use the factory classes in `vision_datasets.common.factory` to create operations for certain data type.

```python
from vision_datasets.common import DatasetTypes, SplitFactory, SplitConfig


data_manifest = ....
splitter = SplitFactory.create(DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, SplitConfig(ratio=0.3))
manifest_1, manifest_2 = splitter.run(data_manifest)
```

### Training with PyTorch

Training with PyTorch is easy. After instantiating a `VisionDataset`, simply passing it in `vision_datasets.common.dataset.TorchDataset` together with the `transform`, then you are good to go with the PyTorch DataLoader for training.


## Helpful commands

There are a few commands that come with this repo once installed, such as datset check and download, detection conversion to classification dataset, and so on, check [`UTIL_COMMANDS.md`](./UTIL_COMMANDS.md) for details.
