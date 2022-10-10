# Introduction

TSV format uses a single or multiple *.tvs format to store both the image annotation and image files, where the first column is the image_id, second column being the annotaion, and third column being the [base64-encoded](https://en.wikipedia.org/wiki/Base64) string of the image data.

This repo does not support consuming TSV format, but we provide tools for converting to/from TSV format from/to coco, for limited tasks and data.

```bash
vision_convert_to_tsv {dataset_name} -r {dataset_registry_json} -k {data storage url} -f {local_dir} [-u Usages]
```

# Task-wise Format

## Image Classifciation

Below is an example of multiclass classification:

```
1   [{"class": "dog"}]    /9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA.....
2   [{"class": "cat"}]    /9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA.....
3   [{"class": "wolff"}]    /9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA.....
...
```

For multilabel classification,

```
1   [{"class": "dog"}, {"class": "canidae"}, {"class": "pet"}]    /9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA.....
2   [{"class": "cat"}, {"class": "Felidae"}, {"class": "pet"}]    /9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA.....
3   [{"class": "wolff"}, {"class": "canidae"}}]    /9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA.....
...
```

## Object Detection

The format of object detection is very similar to mutlilable classification, with an additonal field `rect: [left, top, width, height]`

```
1   [{"class": "dog", "rect": [10, 10, 100, 100]}, {"class": "cat", "rect": [10, 10, 100, 100]}]    /9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA.....
2   [{"class": "cat", "rect": [10, 20, 250, 100]}]    /9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA.....
3   [{"class": "wolff", "rect": [100, 200, 250, 1000]}]    /9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA.....
...
```


## Image Caption


The format of image caption is straightforward as

```
1   [{"caption": "dog playing with a cat"}]    /9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA.....
2   [{"caption": "dog eating food"}]    /9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA.....
3   [{"caption": "wolff sitting in snow"]    /9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA.....
...
```
