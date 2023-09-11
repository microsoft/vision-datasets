import numpy as np
from PIL import Image

from ..common import ImageLabelManifest, FileReader


class ImageMattingLabelManifest(ImageLabelManifest):
    """
    matting: 2D numpy array that has the same width and height with the image
    """

    @property
    def matting_image(self) -> np.ndarray:
        return self.label_data

    def _read_label_data(self):
        file_reader = FileReader()
        with file_reader.open(self.label_path) as f:
            label = np.asarray(Image.open(f))
        file_reader.close()

        return label

    def _check_label(self, label_data):
        if label_data is None:
            raise ValueError
