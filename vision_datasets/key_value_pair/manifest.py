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
    BBOX = 7


def _key_value_pair_value_type_to_enum(val: str):
    return KeyValuePairValueTypes[val.upper()]


class KeyValuePairFieldSchema:
    TYPE_NAME_TO_PYTHON_TYPE = {
        KeyValuePairValueTypes.ARRAY: list,
        KeyValuePairValueTypes.OBJECT: dict,
        KeyValuePairValueTypes.NUMBER: (float, int),
        KeyValuePairValueTypes.INTEGER: int,
        KeyValuePairValueTypes.BOOLEAN: bool,
        KeyValuePairValueTypes.STRING: str, 
        KeyValuePairValueTypes.BBOX: list  # ltrb format in absolute pixel values
    }
    
    def __init__(self, type: str,
                 description: str = None,
                 examples: List[str] = None,
                 enum: List[Union[str, int, float]] = None,
                 items: 'KeyValuePairFieldSchema' = None,
                 properties: Dict[str, 'KeyValuePairFieldSchema'] = None) -> None:
        """
        Key-value pair schema for each field.
        
        Args:
            type (str): type of the field, one of KeyValuePairValueTypes names in lower case.
            description (str): description of the field
            examples (list of str): examples of the field
            enum (list of str/float/int): if the field is restricted to a list of values, define that list. Only work when type is string/number/integer.
            items (KeyValuePairFieldSchema): each item's schema when type is array
            properties (dict of KeyValuePairFieldSchema): properties schema when type is object
        """
        self.type = _key_value_pair_value_type_to_enum(type)
        self.description = description
        self.examples = examples
        self.enum = enum
        self.items = KeyValuePairFieldSchema(**items) if items else None
        self.properties = {k: KeyValuePairFieldSchema(**v) for k, v in properties.items()} if properties else None
        self._check()
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, KeyValuePairFieldSchema):
            return False
        return (self.type == other.type
                and self.description == other.description
                and self.examples == other.examples
                and self.enum == other.enum
                and self.items == other.items
                and self.properties == other.properties)
    
    def _check(self):
        if self.type not in self.TYPE_NAME_TO_PYTHON_TYPE:
            raise ValueError(f'Invalid type: {self.type}')
        if self.enum and self.type not in [KeyValuePairValueTypes.STRING, KeyValuePairValueTypes.NUMBER, KeyValuePairValueTypes.INTEGER]:
            raise ValueError('enum is only allowed for string, number, integer types')
        if self.enum and len(self.enum) != len(set(self.enum)):
            raise ValueError('enum values must be unique')
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
    {
        "key_value_pairs": {"key1": "value1", ...},
        "text": "optional text input for this annotation"
    }
    """
    LABEL_KEY = 'key_value_pairs'
    TEXT_INPUT_KEY = 'text'
    IMAGES_INPUT_KEY = 'image_ids'
    
    @property
    def key_value_pairs(self) -> dict:
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
    def check_schema_match(cls, key_value_pairs: Dict, schema: KeyValuePairSchema):
        for key, field_schema in schema.field_schema.items():
            if key not in key_value_pairs:
                raise ValueError(f'{key} not found')
            KeyValuePairLabelManifest.check_field_schema_match(key_value_pairs[key], field_schema)

    @classmethod
    def check_field_schema_match(cls, value, field_schema: KeyValuePairFieldSchema):
        if not isinstance(value, KeyValuePairFieldSchema.TYPE_NAME_TO_PYTHON_TYPE[field_schema.type]):
            raise ValueError(f'{value} is not of type {field_schema.type}')
        if field_schema.enum:
            if value not in field_schema.enum:
                raise ValueError(f'{value} not in enum {field_schema.enum}')
        if field_schema.type == KeyValuePairValueTypes.BBOX:
            if len(value) != 4 or any(x < 0 for x in value):
                raise ValueError('bbox must have 4 non-negative elements: left, top, right, bottom absolute pixel values!')
        elif field_schema.type == KeyValuePairValueTypes.ARRAY:
            for array_item in value:
                cls.check_field_schema_match(array_item, field_schema.items)
        elif field_schema.type == KeyValuePairValueTypes.OBJECT:
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
