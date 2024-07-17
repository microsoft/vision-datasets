from typing import Optional
from ..common import ImageLabelManifest


class KVPairLabelManifest(ImageLabelManifest):
    """
    {
        "key_value_pairs": {"key1": "value1", ...},
        "text_input": "optional text input for this annotation"
    }
    """
    KV_PAIR_KEY = 'key_value_pairs'
    INPUT_KEY = 'text_input'
    
    @property
    def key_value_pairs(self) -> dict:
        return self.label_data[KVPairLabelManifest.KV_PAIR_KEY]

    @property
    def text_input(self) -> Optional[dict]:
        return self.label_data.get(KVPairLabelManifest.INPUT_KEY, None)

    def _read_label_data(self):
        raise NotImplementedError
    
    def _check_label(self, label_data):
        if KVPairLabelManifest.KV_PAIR_KEY not in label_data:
            raise ValueError    
    