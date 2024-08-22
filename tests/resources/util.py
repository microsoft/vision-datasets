import itertools
import json
import pathlib
import tempfile

from vision_datasets.common import CocoManifestAdaptorFactory, DatasetTypes
TYPES_WITH_CATEGORIES = [DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS, DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL, DatasetTypes.IMAGE_OBJECT_DETECTION]


def coco_dict_to_manifest(task, coco_dict, schema: dict = None):
    if task == DatasetTypes.MULTITASK:
        return coco_dict_to_manifest_multitask(coco_dict[0], coco_dict[1])
    # schema is only required for key_value_pair dataset
    adaptor = CocoManifestAdaptorFactory.create(task, schema) if schema else CocoManifestAdaptorFactory.create(task)
    with tempfile.TemporaryDirectory() as temp_dir:
        dm1_path = pathlib.Path(temp_dir) / 'coco.json'
        dm1_path.write_text(json.dumps(coco_dict))
        return adaptor.create_dataset_manifest(str(dm1_path))


def coco_dict_to_manifest_multitask(tasks, coco_dicts):
    assert len(tasks) == len(coco_dicts)
    task_names = [f'{i}_{task}' for i, task in enumerate(tasks)]
    adaptor = CocoManifestAdaptorFactory.create(DatasetTypes.MULTITASK, {x: y for x, y in zip(task_names, tasks)})
    coco_files = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(len(tasks)):
            dm1_path = pathlib.Path(temp_dir) / f'coco{i}.json'
            dm1_path.write_text(json.dumps(coco_dicts[i]))
            coco_files[task_names[i]] = dm1_path
        return adaptor.create_dataset_manifest(coco_files, temp_dir)


class ImageCaptionTestCases:
    manifest_dicts = [
        {
            "images": [{"id": 1, "file_name": "train_images.zip@1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "caption": "test 1."},
                {"id": 2, "image_id": 2, "caption": "test 2."},
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@3.jpg"},
                       {"id": 2, "file_name": "train_images.zip@4.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "caption": "test 3."},
                {"id": 2, "image_id": 2, "caption": "test 4."},
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@honda.jpg"},
                       {"id": 2, "file_name": "train_images.zip@kitchen.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "caption": "A black Honda motorcycle parked in front of a garage."},
                {"id": 2, "image_id": 1, "caption": "A Honda motorcycle parked in a grass driveway."},
                {"id": 3, "image_id": 1, "caption": "A black Honda motorcycle with a dark burgundy seat."},
                {"id": 4, "image_id": 1, "caption": "Ma motorcycle parked on the gravel in front of a garage."},
                {"id": 5, "image_id": 1, "caption": "A motorcycle with its brake extended standing outside."},
                {"id": 6, "image_id": 2, "caption": "A picture of a modern looking kitchen area.\n"},
                {"id": 7, "image_id": 2, "caption": "A narrow kitchen ending with a chrome refrigerator."},
                {"id": 8, "image_id": 2, "caption": "A narrow kitchen is decorated in shades of white, gray, and black."},
                {"id": 9, "image_id": 2, "caption": "a room that has a stove and a icebox in it"},
                {"id": 10, "image_id": 2, "caption": "A long empty, minimal modern skylit home kitchen."}
            ],
        }]


class ImageMattingTestCases:
    root_path = str(pathlib.Path(__file__).resolve().parent.parent)

    manifest_dicts = [
        {
            "images": [{"id": 1, "file_name": "train_images.zip@image/test_1.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "label": f"{root_path}/image_matting_test_data.zip@mask/test_1.png"}
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@image/test_1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@image/test_2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "label": f"{root_path}/image_matting_test_data.zip@mask/test_1.png"},
                {"id": 2, "image_id": 2, "label": f"{root_path}/image_matting_test_data.zip@mask/test_2.png"},
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@image/test_1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@image/test_2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "label": f"{root_path}/image_matting_test_data.zip@mask/test_1.png"},
                {"id": 2, "image_id": 2, "label": f"{root_path}/image_matting_test_data.zip@mask/test_2.png"},
            ]
        }]


class ImageRegressionTestCases:
    manifest_dicts = [
        {
            "images": [{"id": 1, "file_name": "train_images.zip@1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "target": 1.0},
                {"id": 2, "image_id": 2, "target": 2.0},
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@3.jpg"},
                       {"id": 2, "file_name": "train_images.zip@4.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "target": 3.0},
                {"id": 2, "image_id": 2, "target": 4.0},
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@3.jpg"},
                       {"id": 2, "file_name": "train_images.zip@4.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "target": 5.0},
                {"id": 2, "image_id": 2, "target": 6.0},
            ],
        }]


class ImageTextMatchingTestCases:
    manifest_dicts = [
        {
            "images": [{"id": 1, "file_name": "train_images.zip@1.jpg"},
                       {"id": 2, "file_name": "train_images.zip@2.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "text": "test 1.", "match": 0},
                {"id": 2, "image_id": 2, "text": "test 2.", "match": 0},
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@3.jpg"},
                       {"id": 2, "file_name": "train_images.zip@4.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "text": "test 3.", "match": 0},
                {"id": 2, "image_id": 2, "text": "test 4.", "match": 1},
            ]
        },
        {
            "images": [{"id": 1, "file_name": "train_images.zip@honda.jpg"},
                       {"id": 2, "file_name": "train_images.zip@kitchen.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "text": "A black Honda motorcycle parked in front of a garage.", 'match': 0},
                {"id": 2, "image_id": 1, "text": "A Honda motorcycle parked in a grass driveway.", 'match': 1},
                {"id": 3, "image_id": 1, "text": "A black Honda motorcycle with a dark burgundy seat.", 'match': 1},
                {"id": 4, "image_id": 1, "text": "Ma motorcycle parked on the gravel in front of a garage.", 'match': 0},
                {"id": 5, "image_id": 1, "text": "A motorcycle with its brake extended standing outside.", 'match': 0},
                {"id": 6, "image_id": 2, "text": "A picture of a modern looking kitchen area.\n", 'match': 1},
                {"id": 7, "image_id": 2, "text": "A narrow kitchen ending with a chrome refrigerator.", 'match': 0},
                {"id": 8, "image_id": 2, "text": "A narrow kitchen is decorated in shades of white, gray, and black.", 'match': 0},
                {"id": 9, "image_id": 2, "text": "a room that has a stove and a icebox in it", 'match': 0},
                {"id": 10, "image_id": 2, "text": "A long empty, minimal modern skylit home kitchen.", 'match': 1}
            ],
        }]


class MultiClassClassificationTestCases:
    manifest_dicts = [
        {
            "images": [
                {"id": 1, "width": 224.0, "height": 224.0, "file_name": "train/1.jpg"},
                {"id": 2, "width": 224.0, "height": 224.0, "file_name": "train/3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1},
                {"id": 2, "category_id": 2, "image_id": 2}
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        }
    ]


class MultiLabelClassificationTestCases:
    manifest_dicts = [
        {
            "images": [
                {"id": 1, "width": 224.0, "height": 224.0, "file_name": "train/1.jpg"},
                {"id": 2, "width": 224.0, "height": 224.0, "file_name": "train/3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1},
                {"id": 2, "category_id": 1, "image_id": 2},
                {"id": 3, "category_id": 2, "image_id": 2}
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        },
        {
            "images": [
                {"id": 1, "width": 224.0, "height": 224.0, "file_name": "test/1.jpg"},
                {"id": 2, "width": 224.0, "height": 224.0, "file_name": "test/2.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1},
                {"id": 2, "category_id": 1, "image_id": 2},
                {"id": 3, "category_id": 2, "image_id": 2}
            ],
            "categories": [
                {"id": 1, "name": "tiger"},
                {"id": 2, "name": "rabbit"}
            ]
        },
        {
            "images": [
                {"id": 1, "width": 224.0, "height": 224.0, "file_name": "test/1.jpg"},
                {"id": 2, "width": 224.0, "height": 224.0, "file_name": "test/2.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1},
                {"id": 2, "category_id": 2, "image_id": 2}
            ],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"}
            ]
        }]


class ObjectDetectionTestCases:
    manifest_dicts = [
        {
            "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "siberian-kitten.jpg"},
                       {"id": 2, "width": 224.0, "height": 224.0, "file_name": "kitten 3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1, "bbox": [10, 10, 90, 90]},
                {"id": 2, "category_id": 1, "image_id": 2, "bbox": [100, 100, 100, 100]},
                {"id": 3, "category_id": 2, "image_id": 2, "bbox": [20, 20, 180, 180]}
            ],
            "categories": [
                {"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}
            ]
        },
        {
            "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "siberian-kitten.jpg"},
                       {"id": 2, "width": 224.0, "height": 224.0, "file_name": "kitten 3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1, "bbox": [10, 10, 80, 80]},
                {"id": 2, "category_id": 1, "image_id": 2, "bbox": [90, 90, 90, 90]},
                {"id": 3, "category_id": 2, "image_id": 2, "bbox": [20, 20, 180, 180]}
            ],
            "categories": [
                {"id": 1, "name": "tiger"}, {"id": 2, "name": "rabbit"}
            ]
        },
        {
            "images": [{"id": 1, "width": 224.0, "height": 224.0, "file_name": "siberian-kitten.jpg"},
                       {"id": 2, "width": 224.0, "height": 224.0, "file_name": "kitten 3.jpg"}],
            "annotations": [
                {"id": 1, "category_id": 1, "image_id": 1, "bbox": [10, 10, 80, 80]},
                {"id": 2, "category_id": 2, "image_id": 2, "bbox": [90, 90, 90, 90]},
            ],
            "categories": [
                {"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}
            ]
        }]


class Text2ImageRetrievalTestCases:
    manifest_dicts = [
        {
            "images": [
                {"id": 1, "file_name": "test1.zip@test/0/image_1.jpg"}, {"id": 2, "file_name": "test2.zip@test/1/image_2.jpg"}
            ],
            "annotations": [
                {"image_id": 1, "id": 1, "category_id": 1, "query": "apple"},
                {"image_id": 2, "id": 2, "category_id": 2, "query": "banana"}
            ]
        },
        {
            "images": [
                {"id": 1, "file_name": "test1.zip@test/0/image_1.jpg"}, {"id": 2, "file_name": "test2.zip@test/1/image_2.jpg"}
            ],
            "annotations": [
                {"image_id": 1, "id": 1, "query": "apple"},
                {"image_id": 2, "id": 2, "query": "banana"}
            ]
        }
    ]


class VisualQuestionAnsweringTestCases:
    manifest_dicts = [
        {
            "images": [
                {"id": 1, "file_name": "test1.zip@test/0/image_1.jpg"}, {"id": 2, "file_name": "test2.zip@test/1/image_2.jpg"}
            ],
            "annotations": [
                {"image_id": 1, "id": 1, "question": "what is an apple", "answer": "a kind of fruit"},
                {"image_id": 2, "id": 2, "question": "is apple better than banana", "answer": "no idea"},
            ]
        },
        {
            "images": [
                {"id": 1, "file_name": "test1.zip@test/0/image_1.jpg"}, {"id": 2, "file_name": "test2.zip@test/1/image_2.jpg"}, {"id": 3, "file_name": "test2.zip@test/1/image_3.jpg"}
            ],
            "annotations": [
                {"image_id": 1, "id": 1, "question": "what is apple", "answer": "a kind of fruit"},
                {"image_id": 2, "id": 2, "question": "is apple better than banana", "answer": "no idea"},
                {"image_id": 2, "id": 3, "question": "is banana better than apple", "answer": "stop"},
            ]
        }
    ]


class VisualObjectGroundingTestCases:
    manifest_dicts = [
        {
            "images": [
                {"id": 1, "file_name": "test1.zip@test/0/image_1.jpg"}, {"id": 2, "file_name": "test2.zip@test/1/image_2.jpg"}
            ],
            "annotations": [
                {
                    "image_id": 1,
                    "id": 1,
                    "question": "where are the apples",
                    "answer": "who knows",
                    "groundings": [{"id": 1, "text": "left top corner", "text_span": [0, 1], "bboxes": [[0, 10, 10, 10], [20, 20, 10, 10]]}]},
                {
                    "image_id": 2,
                    "id": 2,
                    "question": "where are the banana",
                    "answer": "check the grounding",
                    "groundings": [{"id": 1, "text": "right bottom corner", "text_span": [0, 1], "bboxes": [[90, 90, 10, 10], [15, 15, 20, 20]]}]
                },
            ]
        },
        {
            "images": [
                {"id": 1, "file_name": "test1.zip@test/0/image_1.jpg"}, {"id": 2, "file_name": "test2.zip@test/1/image_2.jpg"}, {"id": 3, "file_name": "test2.zip@test/1/image_3.jpg"}
            ],
            "annotations": [
                {
                    "image_id": 1,
                    "id": 1,
                    "question": "Describe the image",
                    "answer": "many books",
                    "groundings": [
                            {"id": 1, "text": "20 books", "text_span": [0, 1], "bboxes": [[0, 10, 10, 10], [0, 5, 10, 10]]},
                            {"id": 2, "text": "10 books", "text_span": [2, 4], "bboxes": [[90, 90, 10, 10], [0, 10, 10, 10]]}
                    ]
                },
                {
                    "image_id": 2,
                    "id": 2,
                    "question": "where is an banana",
                    "answer": "present in the image",
                    "groundings": [
                        {"id": 1, "text": "mid of the image", "text_span": [0, 4], "bboxes": [[50, 50, 10, 10], [5, 15, 35, 45]]},
                    ]
                },
                {
                    "image_id": 2,
                    "id": 3,
                    "question": "describe the top half of the image",
                    "answer": "ok",
                    "groundings": [{"id": 1, "text": "Sun rise", "text_span": [0, 1], "bboxes": [[0, 0, 100, 50]]}]
                },
            ]
        }
    ]
 

class KeyValuePairTestCases:
    schema_dicts = [
        {
            "name": "UI automation Schema",
            "description": "Find UI elements and actions by natural language query from given image.",
            "fieldSchema": {
                "action": {
                    "type": "string",
                    "description": "type of action to take on ui element",
                    "classes": {
                        "click": {"description": "click on the element"},
                        "select": {"description": "select the element"},
                        "type": {"description": "type in the element"}
                    },
                    "includeGrounding": True
                }
            }
        },
        {
            "name": "Retail Fraud Detection Schema",
            "description": "Schema for detecting retail fraud by comparing product images",
            "fieldSchema": {
                "productMatch": {
                    "type": "boolean",
                    "description": "Does the product match between the two images"
                },
                "rationale": {
                    "type": "string",
                    "description": "Reason for the 'Product Match' decision"
                },
                "hasDamage": {
                    "type": "boolean",
                    "description": "Is image 2 damaged based on comparison"
                },
                "damageDetails": {
                    "type": "string",
                    "description": "Describe the damage if any in detail"
                }
            }
        },
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
                        "includeGrounding": True
                    }
                }
            }
        }
    ]

    manifest_dicts = [
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
            "bbox_format": "ltrb",
            "annotations": [
                {
                    "id": 1,
                    "image_ids": [1],
                    "text": {
                        "query": "Complete the order"
                    },
                    "fields": {
                        "action": {
                            "value": "click",
                            "groundings": [[50, 50, 100, 100]]
                        }
                    }
                }
            ]
        },
        {
            "images": [
                {
                    "id": 1,
                    "width": 100,
                    "height": 100,
                    "file_name": "0.jpg",
                    "zip_file": "test.zip",
                    "metadata": {
                        "catalog": True,
                        "description": "iphone 12"
                    }
                },
                {
                    "id": 2,
                    "width": 50,
                    "height": 50,
                    "file_name": "1.jpg",
                    "zip_file": "test.zip",
                    "metadata": {
                        "catalog": False,
                        "description": "user 1xxx's review."
                    }
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_ids": [1, 2],
                    "fields": {
                        "productMatch": {"value": False},
                        "rationale": {"value": "The products appear to be similar, but do not have the same brand name or text on them. The catalog image also has more \
                            than one port on the left side and a curved appearance, while the product image has ports on two sides and has a boxy appearance with no curves."},
                        "hasDamage": {"value": True},
                        "damageDetails": {"value": "Scratch on the outside"}
                    }
                },
                {
                    "id": 1,
                    "image_ids": [2, 1],
                    "text": {
                        "note": "reversed image order."
                    },
                    "fields": {
                        "productMatch": {"value": False},
                        "rationale": {"value": ""},
                        "hasDamage": {"value": True},
                        "damageDetails": {"value": "Scratch on the outside"}
                    }
                }
            ]
        },
        {
            "images": [
                {
                    "id": 1,
                    "width": 224,
                    "height": 224,
                    "file_name": "1.jpg",
                    "zip_file": "test.zip"
                },
                {
                    "id": 2,
                    "width": 224,
                    "height": 224,
                    "file_name": "2.jpg",
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
                                {
                                    "value": "scratch",
                                    "groundings": [[10, 10, 10, 10], [30, 30, 10, 10]]
                                },
                                {
                                    "value": "dent",
                                    "groundings": [[80, 80, 20, 20]]
                                }
                            ]
                        }
                    }
                }
            ]
        }
    ]


# Database for valid coco dicts per task
coco_database = {
    DatasetTypes.IMAGE_CAPTION: ImageCaptionTestCases.manifest_dicts,
    DatasetTypes.IMAGE_MATTING: ImageMattingTestCases.manifest_dicts,
    DatasetTypes.IMAGE_REGRESSION: ImageRegressionTestCases.manifest_dicts,
    DatasetTypes.IMAGE_TEXT_MATCHING: ImageTextMatchingTestCases.manifest_dicts,
    DatasetTypes.IMAGE_CLASSIFICATION_MULTICLASS: MultiClassClassificationTestCases.manifest_dicts,
    DatasetTypes.IMAGE_CLASSIFICATION_MULTILABEL: MultiLabelClassificationTestCases.manifest_dicts,
    DatasetTypes.IMAGE_OBJECT_DETECTION: ObjectDetectionTestCases.manifest_dicts,
    DatasetTypes.TEXT_2_IMAGE_RETRIEVAL: Text2ImageRetrievalTestCases.manifest_dicts,
    DatasetTypes.VISUAL_QUESTION_ANSWERING: VisualQuestionAnsweringTestCases.manifest_dicts,
    DatasetTypes.VISUAL_OBJECT_GROUNDING: VisualObjectGroundingTestCases.manifest_dicts,
    DatasetTypes.KEY_VALUE_PAIR: KeyValuePairTestCases.manifest_dicts
}

schema_database = {
    DatasetTypes.KEY_VALUE_PAIR: KeyValuePairTestCases.schema_dicts
}


def two_tasks_test_cases(coco_database):
    tasks = list(coco_database.keys())
    tasks = [t for t in tasks if t != DatasetTypes.KEY_VALUE_PAIR]
    two_tasks = list(itertools.product(tasks, tasks))
    coco_dicts = [list(itertools.product(coco_database[task1], coco_database[task2])) for task1, task2 in two_tasks]
    assert len(two_tasks) == len(coco_dicts)
    tasks_coco_dict = [(two_tasks[i], y) for i, x in enumerate(coco_dicts) for y in x]
    return tasks_coco_dict


coco_database[DatasetTypes.MULTITASK] = two_tasks_test_cases(coco_database)
