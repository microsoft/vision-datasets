# Data Preparation Instruction

## Data format

This repo uses [**Coco format**](https://cocodataset.org/#home) to store the location of the images and the annotations. However, Coco format does not have a rigid requirement about the presence of fields, and it does not cover all tasks.

In [COCO_DATA_FORMAT.md](COCO_DATA_FORMAT.md) doc, we make concrete requirement of what is expected in the Coco json, by this repo/pkg for various tasks, such as image classification, object detection, matting, caption, and so on.

## Data folder structure

A typical folder strucure can look like:

```
standord_cars/
    train_images/
        1.jpg
        2.jpg
        3.jpg
        ...
    test_images/
        1.jpg
        2.jpg
        3.jpg
        ...

    train_images.zip
    test_images.zip
    train_coco.json
    test_coco.json
```

where images are stored under `images/` and `imagez.zip`. For hosting the data on cloud or sharing, **ZIP IS NECESSARY**. Please zip (with **UNCOMPRESSED** aka **STORE** mode) the images into **one** or **multiple** zip files. You can do it with `7zip` (set compression level to 'store') on Windows or [zip](https://superuser.com/questions/411394/zip-files-without-compression) command on Linux. (Note that make sure you do have a `train` folder in your `train.zip`.)

Once zipped, you can specify `"zip_file": "train_images.zip"` in the coco json for each image (check [COCO_DATA_FORMAT.md](COCO_DATA_FORMAT.md).).

### Zip principles

1. **Store train, val and test splits in different zip files.** For example, when doing zero-shot evaluation, all we need is the test split. By zipping them into different zips, we won't have to download a huge zip including training or val images when we just need the test images.
2. **Make each zip file small: <= 5GB**. Given our existing download client, it is more likely for a large file to suffer from network issue or download error, so please keep each zip small. It is ok to have multiple smaller zip files. If you are using your own Azure Blob Client to manage the download or data access, then file size might not be an issue.

### File name and path

Per this [post](https://stackoverflow.com/questions/1976007/what-characters-are-forbidden-in-windows-and-linux-directory-names), below are illegal chars in file paths for Win/Linux:

```
Linux/Unix:
  / (forward slash)

Windows:

  < (less than)
  > (greater than)
  : (colon - sometimes works, but is actually NTFS Alternate Data Streams)
  " (double quote)
  / (forward slash)
  \ (backslash)
  | (vertical bar or pipe)
  ? (question mark)
  * (asterisk)

```

Please avoid those special chars. Even if you are with Linux/Unix, please avoid special chars in Windows as well, to make sure Win users can use your data as well.

## Data check

After everything is done, remember to run the commands below to do a final check, to make sure your data is ready for check in
- `pip install vision-datasets -U`
- Taking the stanford_cars as an example:
  - `vision_check_dataset stanford_cars -c train_coco.json -f ./ -t classification_multiclass`
  - `vision_check_dataset stanford_cars -c test_coco.json -f ./ -t classification_multiclass`

Support for `key_value_pair` dataset is coming soon.

## Host/manage datasets on cloud/disk

`DatasetHub` class is the one that manages access of multiple datasets either from local disk or cloud, using the dataset name and version information. It takes a dataset regisration json file, which contains the meta information of each dataset. For each dataset, there is a corresponding entry in the json.

Below are two examples of single-task datasets:

```{json}
[
    {
        "name": "sampled-ms-coco",
        "version": 1,
        "description": "A sampled ms-coco dataset.",
        "type": "object_detection",
        "format": "coco", // indicating the annotation data are stored in coco format
        "root_folder": "detection/coco2017_20200401", // a root folder for all files listed
        "train": {
            "index_path": "train.json", // coco json file for training, see next section for example
            "files_for_local_usage": [ "images/train_images.zip" ] // associated files including data such as images 
        },
        "test": {
            "index_path": "test.json",
            "files_for_local_usage": [ "images/test_images.zip" ]
        }
    },
    {
        "name": "stanford-cars",
        "description": "The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.",
        "version": 1,
        "format": "coco",
        "type": "classification_multiclass",
        "root_folder": "classification/stanford_cars_20211007",
        "train": {
            "index_path": "train_coco.json",
            "files_for_local_usage": [ "train_images.zip" ],
        },
        "test": {
            "index_path": "test_coco.json",
            "files_for_local_usage": [ "test_images.zip" ],
        }
    },
]
```

Below is an example of multitask dataset:

```json
[
    {
        "name": "people-dataset",
        "description": "people dataset including gender, height, and location information",
        "version": 1,
        "format": "coco",
        "type": "multitask",
        "root_folder": "multitask/people_dataset",
        "tasks": {
            "gender":{
                "type": "classification_multiclass",
                "train": {
                    "index_path": "train_coco_gender.json", "files_for_local_usage": [ "train_images.zip" ],
                },
                "test": {
                    "index_path": "test_coco_gender.json", "files_for_local_usage": [ "test_images.zip" ],
                }
            },
            "height":{
                "type": "image_regression",
                "train": {
                    "index_path": "train_coco_height.json", "files_for_local_usage": [ "train_images.zip" ],
                },
                "test": {
                    "index_path": "test_coco_height.json", "files_for_local_usage": [ "test_images.zip" ],
                }
            },
            "location":{
                "type": "object_detection",
                "train": {
                    "index_path": "train_coco_location.json", "files_for_local_usage": [ "train_images.zip" ],
                },
                "test": {
                    "index_path": "test_coco_location.json", "files_for_local_usage": [ "test_images.zip" ],
                }
            },
        }
    }
]
```

For `key_value_pair` dataset, an additional field `schema` is required to define the task and the label format. Below is an example:

```json
[
    {
        "name": "multi-image-question-answer",
        "version": 1,
        "description": "Answer question related to one or more images",
        "type": "key_value_pair",
        "format": "coco",
        "root_folder": "kv_pair/multi_img_qa_20240723",
        "schema": {
        "name": "Multi-image QA schema",
        "description": "Provide answer and a rationale.",
        "fieldSchema": {
            "answer": {
            "type": "string",
            "description": "answer."  
            },
            "rationale": {
            "type": "string",
            "description" :"rationale of answer"
            }
        }    
        },
        "train": {
            "index_path": "train.json",
            "files_for_local_usage": [
                "train_images.zip"
            ]
        },
        "val": {
            "index_path": "val.json",
            "files_for_local_usage": [
                "val_images.zip"
            ]
        }
    }
]
```

Check the usage code example in [`README.md`](README.md).
