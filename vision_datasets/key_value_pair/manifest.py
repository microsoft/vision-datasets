from typing import Dict, List, Optional, Union
from enum import Enum
from ..common import MultiImageLabelManifest, DatasetManifestWithMultiImageLabel, DatasetTypes


class KeyValuePairValueTypes(Enum):
    ARRAY = 1
    OBJECT = 2
    NUMBER = 3
    INTEGER = 4
    BOOLEAN = 5
    STRING = 6


def _key_value_pair_value_type_to_enum(val: str):
    return KeyValuePairValueTypes[val.upper()]


def _valid_ltrb_bbox(bbox: List) -> bool:
    # bbox must have 4 non-negative elements: left, top, right, bottom absolute pixel values.
    return (len(bbox) == 4 and all(x >= 0 for x in bbox) and bbox[2] > bbox[0] and bbox[3] > bbox[1])


class KeyValuePairClassSchema:
    def __init__(self, description: str = None, examples: List = None) -> None:
        self.description = description
        
    def __eq__(self, other) -> bool:
        if not isinstance(other, KeyValuePairClassSchema):
            return False
        return self.description == other.description


class KeyValuePairFieldSchema:
    TYPE_NAME_TO_PYTHON_TYPE = {
        KeyValuePairValueTypes.ARRAY: list,
        KeyValuePairValueTypes.OBJECT: dict,
        KeyValuePairValueTypes.NUMBER: (float, int),
        KeyValuePairValueTypes.INTEGER: int,
        KeyValuePairValueTypes.BOOLEAN: bool,
        KeyValuePairValueTypes.STRING: str
    }

    def __init__(self, type: str,
                 description: str = None,
                 examples: List[str] = None,
                 classes: Dict[Union[str, int, float], KeyValuePairClassSchema] = None,
                 items: 'KeyValuePairFieldSchema' = None,
                 properties: Dict[str, 'KeyValuePairFieldSchema'] = None,
                 includeGrounding: bool = False) -> None:
        """
        Key-value pair schema for each field.
        The annotation of each key is a dictionary containing
            1. "value"  field that contains the annotation
            2. "groundings" field enabled when includeGrounding=True, which contains a list of grounded bounding boxes in the image for the annotation

        Args:
            type (str): type of the field, one of KeyValuePairValueTypes names in lower case.
            description (str): description of the field
            examples (list of str): examples of the field
            classes (dict[str|int|float, KeyValuePairClassSchema]): if the field is restricted to a list of classes, defines the map from class name to its information. Only works when type is string.
            items (KeyValuePairFieldSchema): each item's schema when type is array
            properties (dict of KeyValuePairFieldSchema): properties schema when type is object,
            includeGrounding (bool): whether the field should be grounded a list of bboxes in the image.
        """
        self.type = _key_value_pair_value_type_to_enum(type)
        self.description = description
        self.examples = examples
        self.classes = {k: KeyValuePairClassSchema(v) for k, v in classes.items()} if classes else None
        self.items = KeyValuePairFieldSchema(**items) if items else None
        self.properties = {k: KeyValuePairFieldSchema(**v) for k, v in properties.items()} if properties else None
        self.includeGrounding = includeGrounding
        self._check()

    def __eq__(self, other) -> bool:
        if not isinstance(other, KeyValuePairFieldSchema):
            return False
        return (self.type == other.type
                and self.description == other.description
                and self.examples == other.examples
                and self.classes == other.classes
                and self.items == other.items
                and self.properties == other.properties
                and self.includeGrounding == other.includeGrounding)

    def _check(self):
        if self.type not in self.TYPE_NAME_TO_PYTHON_TYPE:
            raise ValueError(f'Invalid type: {self.type}')
        if self.classes:
            if self.type not in {KeyValuePairValueTypes.STRING, KeyValuePairValueTypes.INTEGER, KeyValuePairValueTypes.NUMBER}:
                raise ValueError('"classes" is only allowed for string, integer, number types')
            if any(not isinstance(k, str) for k in self.classes.keys()):
                raise ValueError('"classes" keys must be string')
        if self.type == KeyValuePairValueTypes.ARRAY and not self.items:
            raise ValueError('items must be provided for array type')
        elif self.type == KeyValuePairValueTypes.OBJECT and not self.properties:
            raise ValueError('properties must be provided for object type')


class KeyValuePairSchema:
    def __init__(self, name: str, field_schema_dict: Dict, description: str = None) -> None:
        self.name = name
        self.description = description
        self.field_schema = {k: KeyValuePairFieldSchema(**v) for k, v in field_schema_dict.items()}

    def __eq__(self, other) -> bool:
        return isinstance(other, KeyValuePairSchema) and self.name == other.name and self.field_schema == other.field_schema and self.description == other.description


class KeyValuePairLabelManifest(MultiImageLabelManifest):
    """
    Label manifest for key-value pair annotations. The "fields" field follows KeyValuePairSchema.
    For example, the label data can be:
    {
        "fields": {
            "key1": {"value": "v1", "groundings": [[10,10,5,5]]},
            "key2": {"value": "v2"},
            ...
        },
        "text": "optional text input for this annotation"
    },
    the fields follow the schema:
    {
        "name": "example key value pair field schema",
        "fieldSchema": {
            "key1": {"type": "string", "includeGrounding": true},
            "key2": {"type": "string"},
            ...
        }
    }
    """
    LABEL_KEY = 'fields'
    LABEL_VALUE_KEY = 'value'
    LABEL_GROUNDINGS_KEY = 'groundings'
    TEXT_INPUT_KEY = 'text'
    IMAGES_INPUT_KEY = 'image_ids'

    @property
    def fields(self) -> dict:
        return self.label_data[self.LABEL_KEY]

    @property
    def text(self) -> Optional[dict]:
        return self.label_data.get(self.TEXT_INPUT_KEY, None)

    def _read_label_data(self):
        raise NotImplementedError('Read label data is not supported!')

    def _check_label(self, label_data):
        if not isinstance(label_data, dict) or self.LABEL_KEY not in label_data:
            raise ValueError(f'{self.LABEL_KEY} not found in label_data dictionary: {label_data}')

    @classmethod
    def check_schema_match(cls, fields: Dict[str, Dict], schema: KeyValuePairSchema):
        for key, field_schema in schema.field_schema.items():
            if key not in fields:
                raise ValueError(f'{key} not found')
            KeyValuePairLabelManifest.check_field_schema_match(fields[key], field_schema)

    @classmethod
    def check_field_schema_match(cls, value, field_schema: KeyValuePairFieldSchema):
        if not isinstance(value, dict) or cls.LABEL_VALUE_KEY not in value:
            raise ValueError(f'{value} must be a dictionary that maps "{cls.LABEL_VALUE_KEY}" to the annotation.')
        if field_schema.includeGrounding:
            if cls.LABEL_GROUNDINGS_KEY not in value:
                raise ValueError(f'{cls.LABEL_GROUNDINGS_KEY} is required in schema, but not found in {value}.')
            if any(not _valid_ltrb_bbox(bbox) for bbox in value[cls.LABEL_GROUNDINGS_KEY]):
                raise ValueError(f'Invalid bboxes: {value[cls.LABEL_GROUNDINGS_KEY]}. bbox must have 4 non-negative elements: left, top, right, bottom absolute pixel values.')

        value = value[cls.LABEL_VALUE_KEY]
        if not isinstance(value, KeyValuePairFieldSchema.TYPE_NAME_TO_PYTHON_TYPE[field_schema.type]):
            raise ValueError(f'{value} is not of type {field_schema.type}')
        if field_schema.classes and value not in field_schema.classes:
            raise ValueError(f'{value} not found in classes {field_schema.classes}')
        elif field_schema.type == KeyValuePairValueTypes.ARRAY:
            if not isinstance(value, list):
                raise ValueError(f'{value} is not a list')
            for array_item in value:
                cls.check_field_schema_match(array_item, field_schema.items)
        elif field_schema.type == KeyValuePairValueTypes.OBJECT:
            if not isinstance(value, dict):
                raise ValueError(f'{value} is not a dictionary')
            for k, v in value.items():
                if k not in field_schema.properties:
                    raise ValueError(f'{k} not found in schema')
                cls.check_field_schema_match(v, field_schema.properties[k])


class KeyValuePairDatasetManifest(DatasetManifestWithMultiImageLabel):
    """Manifest that has schema in additional_info which defines the structure of the key-value pairs in the annotations."""

    def __init__(self, images, annotations, schema, additional_info):
        self.schema = KeyValuePairSchema(schema['name'], schema['fieldSchema'], schema.get('description'))
        super().__init__(images, annotations, DatasetTypes.KEY_VALUE_PAIR, additional_info)
        self._check_annotations()

    def _check_annotations(self):
        for ann in self.annotations:
            if not isinstance(ann, KeyValuePairLabelManifest):
                raise ValueError(f'label must be of type {KeyValuePairLabelManifest.__name__}')
