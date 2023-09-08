import pathlib
import tempfile
import unittest

import numpy
from PIL import Image, ImageChops

from vision_datasets.common import Base64Utils


class TestBase64Utils(unittest.TestCase):
    @staticmethod
    def _create_rand_img(size=(100, 100)) -> Image.Image:
        assert len(size) == 2

        imarray = numpy.random.rand(size[0], size[1], 3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        return im

    def test_b64_to_file_loses_no_info(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = pathlib.Path(temp_dir)
            img_filepath_1 = temp_dir / 'temp_1.jpg'
            TestBase64Utils._create_rand_img().save(img_filepath_1)

            img_filepath_2 = temp_dir / 'temp_2.jpg'
            b64str = Base64Utils.file_to_b64_str(img_filepath_1)
            Base64Utils.b64_str_to_file(b64str, img_filepath_2)
            img1 = Image.open(img_filepath_1)
            img2 = Image.open(img_filepath_2)
            assert not ImageChops.difference(img1, img2).getbbox()
