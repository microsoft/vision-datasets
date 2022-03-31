# Coco format

## Image classification (multiclass and multilabel)

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

## Object detection

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

- Note that `ltrb` used to be default. If your coco annotations were prepared to work with this repo before version 0.1.2. Please add `"bbox_format": "ltrb"` to your coco file.
- Regardless of what format bboxes are stored in Coco file, when annotations are transformed into `ImageDataManifest`, the bbox will be unified into `ltrb: [left, top, right, bottom]`.

## Image caption

Here is one example of the json file for image caption task.

``` {json}
{
  "images": [{"id": 1, "file_name": "train_images.zip@honda.jpg"},
              {"id": 2, "file_name": "train_images.zip@kitchen.jpg"}],
  "annotations": [
      {"id": 1, "image_id": 1, "caption": "A black Honda motorcycle parked in front of a garage."},
      {"id": 2, "image_id": 1, "caption": "A Honda motorcycle parked in a grass driveway."},
      {"id": 3, "image_id": 2, "caption": "A black Honda motorcycle with a dark burgundy seat."},
  ],
}
```

## Image text matching

Here is one example of the json file for image text matching task. `match: 1` indicates image and text match.

``` {json}
{
  "images": [{"id": 1, "file_name": "train_images.zip@honda.jpg"},
              {"id": 2, "file_name": "train_images.zip@kitchen.jpg"}],
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

Specifically, **only** image files are supported for the label file.

``` {json}
{
    "images": [{"id": 1, "file_name": "train_images.zip@image/test_1.jpg"},
                {"id": 2, "file_name": "train_images.zip@image/test_2.jpg"}],
    "annotations": [
        {"id": 1, "image_id": 1, "label": "image_matting_label.zip@mask/test_1.png"},
        {"id": 2, "image_id": 2, "label": "image_matting_label.zip@mask/test_2.png"},
    ]
}
```
