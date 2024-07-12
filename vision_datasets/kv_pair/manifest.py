from ..common import ImageLabelManifest

class KVPairLabelManifest(ImageLabelManifest):
    """
    {
        "key_value_pairs": {"key1": "value1", ...},
        "text_input": "optional text input for this annotation"
    }
    """
    
    @property
    def key_value_pairs(self):
        return self.label_data['key_value_pairs']

    @property
    def text_input(self):
        return self.label_data['text_input']
