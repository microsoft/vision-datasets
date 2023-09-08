import base64
import io
import pathlib
from typing import Union

from .data_reader import FileReader, PILImageLoader


class Base64Utils:
    @staticmethod
    def b64_str_to_pil(img_b64_str: str):
        assert img_b64_str

        return PILImageLoader.load_from_stream(io.BytesIO(base64.b64decode(img_b64_str)))

    @staticmethod
    def file_to_b64_str(filepath: pathlib.Path, file_reader=None):
        assert filepath

        fr = file_reader or FileReader()
        with fr.open(filepath.as_posix(), "rb") as file_in:
            return base64.b64encode(file_in.read()).decode('utf-8')

    @staticmethod
    def b64_str_to_file(b64_str: str, file_name: Union[pathlib.Path, str]):
        assert b64_str
        assert file_name

        with open(file_name, 'wb') as file_out:
            file_out.write(base64.b64decode(b64_str))
