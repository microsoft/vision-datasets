# Coco format

In coco, we use `file_name` and `zip_file` to construct the file_path in `ImageDataManifest` mentioned in `README.md`. If `zip_file` is present, it means that the image is zipped into a zip file for storage & access, and the path within the zip is `file_name`. If `zip_file` is not present, the image path would just be `file_name`.

## Image classification (multiclass and multilabel)

Here is one example of the train.json, val.json, or test.json in the `DatasetInfo` above. Note that the `"id"` for `images`, `annotations` and `categories` should be consecutive integers, **starting from 1**. Note that our lib might work with id starting from 0, but many tools like [CVAT](https://github.com/openvinotoolkit/cvat/issues/2085) and official [COCOAPI](https://github.com/cocodataset/cocoapi/issues/507) will fail.

```json
{
  "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "train_images/siberian-kitten.jpg", "zip_file": "train_images.zip"},
              {"id": 2, "width": 224.0, "height": 224.0, "file_name": "train_images/kitten 3.jpg", "zip_file": "train_images.zip"}],
              //  file_name is the image path, which supports three formats as described in previous section.
  "annotations": [
      {"id": 1, "category_id": 1, "image_id": 1},
      {"id": 2, "category_id": 1, "image_id": 2},
      {"id": 3, "category_id": 2, "image_id": 2}
  ],
  "categories": [{"id": 1, "name": "cat", "supercategory": "animal"}, {"id": 2, "name": "dog", "supercategory": "animal"}]
}
```

## Object detection

```json
{
  "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "train_images/siberian-kitten.jpg", "zip_file": "train_images.zip"},
              {"id": 2, "width": 224.0, "height": 224.0, "file_name": "train_images/kitten 3.jpg", "zip_file": "train_images.zip"}],
  "annotations": [
      {"id": 1, "category_id": 1, "image_id": 1, "bbox": [10, 10, 100, 100]},
      {"id": 2, "category_id": 1, "image_id": 2, "bbox": [100, 100, 200, 200]},
      {"id": 3, "category_id": 2, "image_id": 2, "bbox": [20, 20, 200, 200], "iscrowd": 1}
  ],
  "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
}
```

You might notice that for the 3rd box, there is a "iscrowd" field. It specifies whether the box is about a crowd of objects.

### BBox Format

bbox format should be **absolute** pixel position following either `ltwh: [left, top, width, height]` or `ltrb: [left, top, right, bottom]`. `ltwh` is the default format. To work with `ltrb`, please specify `bbox_format` to be `ltrb` in coco json file.

```json
{
  "bbox_format": "ltrb",  
  "images": ...,
  "annotations": ...,
  "categories": ...
}
```

Note that

- Note that `ltrb` used to be default. If your coco annotations were prepared to work with this repo before version 0.1.2. Please add `"bbox_format": "ltrb"` to your coco file.
- Regardless of what format bboxes are stored in Coco file, when annotations are transformed into `ImageDataManifest`, the bbox will be unified into `ltrb: [left, top, right, bottom]`.

## Image caption

Here is one example of the json file for image caption task.

```json
{
  "images": [{"id": 1, "file_name": "train_images/honda.jpg", "zip_file": "train_images.zip"},
              {"id": 2, "file_name": "train_images/kitchen.jpg", "zip_file": "train_images.zip"}],
  "annotations": [
      {"id": 1, "image_id": 1, "caption": "A black Honda motorcycle parked in front of a garage."},
      {"id": 2, "image_id": 1, "caption": "A Honda motorcycle parked in a grass driveway."},
      {"id": 3, "image_id": 2, "caption": "A black Honda motorcycle with a dark burgundy seat."},
  ],
}
```

## Image text matching

Here is one example of the json file for image text matching task. `match` is a float between [0, 1], where 0 means not match at all, 1 means perfect match

```json
{
  "images": [{"id": 1, "file_name": "train_images/honda.jpg", "zip_file": "train_images.zip"},
              {"id": 2, "file_name": "train_images/kitchen.jpg", "zip_file": "train_images.zip"}],
  "annotations": [
      {"id": 1, "image_id": 1, "text": "A black Honda motorcycle parked in front of a garage.", "match": 0},
      {"id": 2, "image_id": 1, "text": "A Honda motorcycle parked in a grass driveway.", "match": 0},
      {"id": 3, "image_id": 2, "text": "A black Honda motorcycle with a dark burgundy seat.", "match": 1},
  ],
}
```

## Image matting

Here is one example of the json file for image matting task. The "label" in the "annotations" can be one of the following formats:

- a local path to the label file
- a local path in a non-compressed zip file (`c:\foo.zip@bar.png`)
- a url to the label file

Specifically, **only** image files are supported for the label files. The ground truth image should be one channel image (i.e. `PIL.Image` mode "L", instead of "RGB") that has the same width and height with the image file. Refer to the images in [tests/image_matting_test_data.zip](tests/image_matting_test_data.zip) as an example.

```json
{
    "images": [{"id": 1, "file_name": "train_images/image/test_1.jpg", "zip_file": "train_images.zip"},
                {"id": 2, "file_name": "train_images/image/test_2.jpg", "zip_file": "train_images.zip"}],
    "annotations": [
        {"id": 1, "image_id": 1, "label": "image_matting_label/mask/test_1.png", "zip_file": "image_matting_label.zip"},
        {"id": 2, "image_id": 2, "label": "image_matting_label/mask/test_2.png", "zip_file": "image_matting_label.zip"},
    ]
}
```

## Visual Question Answering

VQA represents the problem where one asks a question about an image and a ground truth answer is associated.

```json
{
    "images": [
        {"id": 1, "zip_file": "test1.zip", "file_name": "test/0/image_1.jpg"},
        {"id": 2, "zip_file": "test2.zip", "file_name": "test/1/image_2.jpg"}
    ],
    "annotations": [
        {"image_id": 1, "id": 1, "question": "what animal is in the image?", "answer": "a cat"},
        {"image_id": 2, "id": 2, "question": "What is the title of the book on the shelf?", "answer": "How to make bread"}
    ]
}
```

## Visual Object Grounding

Visual Object Grounding is a problem where a text query/question about an image is provided, and an answer/caption about the image along with the most relevant grounding(s) are returned.

A grounding is composed of three parts:

- `bbox`: bounding box around the region of interest, same with object detection task. Similarly, you can specify `ltrb` or `ltwh` (default) in the Coco json. Regardlessly, the label manifest will store the bbox in [left, top, right, bottom] format like object detection.
- `text`: description about the region
- `text_span`: two ints (start-inclusive, end-exclusive), indicating the section of text that the region is relevant to in the answer/caption

```json
{
    "images": [
        {"id": 1, "zip_file": "test1.zip", "file_name": "test/0/image_1.jpg"},
        {"id": 2, "zip_file": "test2.zip", "file_name": "test/1/image_2.jpg"}
    ],
    "annotations": [
        {
            "image_id": 1,
            "id": 1,
            "question": "whats animal are in the image?",
            "answer": "cat and bird",
            "groundings": [
                {"text": "a cat", "text_span": [0, 2], "bboxes": [[10, 10, 100, 100], [15, 15, 100, 100]]},
                {"text": "a bird", "text_span": [3, 4], "bboxes": [[15, 15, 30, 30], [0, 10, 20, 20]]}
            ]
        },
        {
            "image_id": 2,
            "id": 2,
            "question": "What is the title and auther of the book on the shelf?",
            "answer": "Tile is baking and auther is John",
            "groundings": [
                {"text": "Title: Baking", "text_span": [0, 2], "bboxes": [[10, 10, 100, 100]]},
                {"text": "Author: John", "text_span": [3, 4], "bboxes": [[0, 0, 50, 50], [15, 15, 25, 25]]}
            ]
        }
    ]
}
```

## Image regression

Here is one example of the json file for the image regression task, where the "target" in the "annotations" field is a real-valued number (e.g. a score, an age, etc.). Note that each image should only have one regression target (i.e. there should be exactly one annotation for each image).

```json
{
    "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "train_images/image_1.jpg", "zip_file": "train_images.zip"},
              {"id": 2, "width": 224.0, "height": 224.0, "file_name": "train_images/image_2.jpg", "zip_file": "train_images.zip"}],
    "annotations": [
        {"id": 1, "image_id": 1, "target": 102.0},
        {"id": 2, "image_id": 2, "target": 28.5}
    ]
}
```

## Image retrieval

This task will be a pure representation of the data of images retrieved by text queries only.

```json
{
    "images": [
        {"id": 1, "zip_file": "test1.zip", "file_name": "test/0/image_1.jpg"},
        {"id": 2, "zip_file": "test2.zip", "file_name": "test/1/image_2.jpg"}
    ],
    "annotations": [
        {"image_id": 1, "id": 1, "query": "Men eating a banana."},
        {"image_id": 2, "id": 2, "query": "An apple on the desk."}
    ]
}
```

## MultiTask dataset

Multitask dataset represents the kind of dataset, where a single set of images possesses multiple sets of annotations for different tasks of single/mutiple tasks mentioned above.

For example, a set of people images can have different attributes: gender/classification {make, female, other}, height/regression: {0-300cm}, person location/detection: {x, y, w, h}, etc.

To represent this kind of dataset, it is simple: create one independent coco file for each task:

```
people_dataset/
    train_images/
        ...
    test_images/
        ...

    train_images.zip
    test_images.zip
    
    train_coco_gender.json
    test_coco_gender.json
    train_coco_height.json
    test_coco_height.json
    train_coco_location.json
    test_coco_location.json
```

## KeyValuePair dataset

It is a generic dataset format, that is annotation-oriented, where data entity of each annotation could involve multiple images and texts. The labeling of each annotation is represented as a dictionary, where keys are the fields of interests related to the data.
Each dataset is associated with a schema to define the task, fields of interests and format of those fields. The schema format follows JSON Schema stype, and is defined below:

| Property    | Type                                       | Details                                                                                                                      | Required?                               |
| :---------- | :----------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------- |
| name        | string                                     | schema name                                                                                                                  | yes                                     |
| description | string                                     | detailed description of the schema. e.g. Extract defect location and type from an image of metal screws on an assembly line. | no, but strongly recommended to provide |
| fieldSchema | dict[string\|number\|integer, FieldSchema] | schemas of fields                                                                                                            | yes                                     |

The schema of each field is defined by `FieldSchema`, recursively:

| Property         | Type                        | Details                                                                                                                                                                                                                   | Required?               |
| :--------------- | :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------- |
| type             | `FieldValueType`            | JSON type: string, number, integer, boolean, array, object.                                                                                                                                                               | yes                     |
| description      | string                      | describes the field in more detail,                                                                                                                                                                                       | no                      |
| examples         | list[string]                | examples of field content,                                                                                                                                                                                                | no                      |
| classes          | dict[str, `ClassSchema`]    | [string-only] dictionary that maps each class name to `ClassSchema`.                                                                                                                                                      | no                      |
| properties       | dict[string, `FieldSchema`] | [object-only] defines FieldSchema of each subfield,                                                                                                                                                                       | yes when type is object |
| items            | `FieldSchema`               | [array-only] defines the FieldSchema for all items in array,                                                                                                                                                              | yes when type is array  |
| includeGrounding | boolean                     | whether annotation of this field has bbox groundings associated; if true, bboxes are stored in the `groundings` field of the annotation. bboxes follow [BBox Format](#bbox-format). Only support single-image annotation. | No, default false       |

Definition of `ClassSchema`:

| Property    | Type   | Details                                                          | Required?         |
| :---------- | :----- | :--------------------------------------------------------------- | :---------------- |
| description | string | description of the class, e.g., "long, thin, surface-level mark" | no. Default: null |

For example, a visual question answering task labeling schema is:

```json
{
  "name": "Visual question answering",
  "description": "Answer questions on given images and provide rationales.",
  "fieldSchema": {
    "answer": {
      "type": "string",
      "description": "Answer to the question."
    },
    "rationale": {
        "type": "string", 
        "description": "Rationale of the answer."
    }
  }
}
```

The fields of interests are `answer` and `rationale`.

In addition, a defect detection schema can be defined as

```json
{
  "name": "Defect detection - screws",
  "description": "Extract defect location and type from an image of metal screws on an assembly line",
  "fieldSchema": {
    "defects": {
      "type": "array",
      "description": "The defect types with bounding boxes detected in the image",
      "items": {
        "type": "string",
        "description": "The type of defect detected",
        "classes": {
          "scratch": {"description": "long, thin, surface-level mark"},
          "dent": {"description": "appears to be caving in"},
          "discoloration": {"description": "color is abnormal"},
          "crack": {"description": "deeper mark than a scratch"}
        },
        "includeGrounding": true
      }
    }
  }
}
```

We can see it is an object detection task with four classes: scratch, dent, discoloration, crack.

More examples can be found at [DATA_PREPARATION.md](DATA_PREPARATION.md). More details can be found at [`vision-datasets/vision_datasets/key_value_pair/manifest.py`](vision_datasets/key_value_pair/manifest.py).

Once schema is defined, we can construct the dataset. In details, each sample consists of:

- data entities:
  - images, image is optionally associated with a metadata dictionary which stores the text attributes of interest for the image. For example, image is a product catalog image: `{'metadata': {'catalog': true}}`, capture location of an image: `{'metadata': {'location': 'street'}}`, information of the assembly component captured in image of a defect detection dataset: `{'metadata': {'name': 'Hex Head Lag Screw', 'type': '3/8-inch x 4-inch'}}`  
  - text (optional), a dictionary with keys being field names e.g. `{'text': {'question': 'a specific question related to the images input'}}`
- labeling involving fields of interests:
  - fields, a dictionary with keys being the **fields of interest**, values being dictionaries that store the actual field value in "value" and optionally a list of grounded bboxes in "groundings". "groundings" are for single-image annotation only. Each bbox follows [BBox Format](#bbox-format). The format of each field should comply to the defined `fieldSchema`.

The dataset format is a simple variation of COCO, where `image_id` of an annotation entry is replaced with `image_ids` to support multi-image annotation.
In each annotation entry, `fields` is required, `text` is optional. In each image entry, `metadata` is optional, for additional image-specific information. Below is an example of multi-image question answering.

```json
{
    "images": [
        {"id": 1, "zip_file": "test1.zip", "file_name": "test/0/image_1.jpg", "metadata": {"location": "street"}},
        {"id": 2, "zip_file": "test2.zip", "file_name": "test/1/image_2.jpg"}
    ],
    "annotations": [
        {
            "id": 1,
            "image_ids": [1, 2],
            "text": {"question": "What objects are unique in the first image compared to the second image?"},
            "fields": {
                "answer": {"value": "car"}, 
                "rationale": {"value": "Both images capture street traffic, a car exists in the first image but not in the second."}
            }
        },
        {
            "id": 2,
            "image_ids": [2, 1],
            "text": {"question": "Does the first image have more cars?"}, 
            "fields": {
                "answer": {"value": "yes"}, 
                "rationale": {"value": "First image has no car, second image has one."}
            }
        }
    ]
}
```

Another example for object detection:

```json
{
    "images": [
        {
            "id": 1,
            "width": 224,
            "height": 224,
            "file_name": "1.jpg",
            "zip_file": "test.zip"
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_ids": [1],
            "fields": {
                "defects": {
                    "value": [
                        {"value": "scratch", "groundings": [[10, 10, 10, 10], [30, 30, 10, 10]]},
                        {"value": "dent", "groundings": [[80, 80, 20, 20]]}
                    ]
                }
            }
        }
    ]
}
```
