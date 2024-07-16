from typing import Optional
from ..common import ImageLabelManifest


class KVPairLabelManifest(ImageLabelManifest):
    """
    {
        "key_value_pairs": {"key1": "value1", ...},
        "text_input": "optional text input for this annotation"
    }
    """

    @property
    def key_value_pairs(self) -> dict:
        return self.label_data['key_value_pairs']

    @property
    def text_input(self) -> Optional[dict]:
        return self.label_data.get('text_input', None)

    def _read_label_data(self):
        raise NotImplementedError
    
    def _check_label(self, label_data):
        if 'key_value_pairs' not in label_data:
            raise ValueError    
    