
import json

import numpy as np
from PIL import Image

from ...data_manifest import ImageLabelManifest
from ...data_reader import FileReader


class ImageMattingLabelManifest(ImageLabelManifest):
    """
    matting: 2D numpy array that has the same width and height with the image
    """

    def _read_label_data(self):
        file_reader = FileReader()
        with file_reader.open(self.label_path) as f:
            label = np.asarray(Image.open(f))
        file_reader.close()

        return label

    def __str__(self) -> str:
        data_dict = self.__getstate__()
        del data_dict['_label_data']  # not serializable
        return f'Label: {json.dumps(data_dict)}'
