from abc import abstractmethod
import pathlib
import typing

from ...base64_utils import Base64Utils
from ...data_reader.file_reader import FileReader
from ..data_manifest import DatasetManifest, ImageDataManifest, ImageLabelManifest
from .operation import Operation


class GenerateStandAloneImageListBase(Operation):
    """
    Base class for generating an image oriented dictonary where each entry contains all information about the image including the image data, annotation, etc...
    """

    def __init__(self, flatten: bool) -> None:
        super().__init__()
        self._flatten = flatten

    def run(self, *args) -> typing.Generator:
        if len(args) != 1:
            raise ValueError

        manifest = args[0]
        file_reader = FileReader()
        if self._flatten:
            for i, image in enumerate(manifest.images):
                b64_image = Base64Utils.file_to_b64_str(pathlib.Path(image.img_path), file_reader=file_reader)
                for label in image.labels:
                    img = {
                        'image_id': i + 1,
                    }
                    label = self._generate_label(label, image, manifest)
                    if isinstance(label, dict):
                        img.update(label)
                    else:
                        img['label'] = label

                    img['image'] = b64_image
                    yield img
        else:
            for i, x in enumerate(manifest.images):
                yield {
                    'id': i + 1,
                    'labels': list(self._generate_labels(x, manifest)),
                    'image': Base64Utils.file_to_b64_str(pathlib.Path(x.img_path), file_reader=file_reader),
                }

    def _generate_labels(self, image: ImageDataManifest, manifest: DatasetManifest) -> typing.Generator:
        for label in image.labels:
            yield self._generate_label(label, image, manifest)

    @abstractmethod
    def _generate_label(self, label: ImageLabelManifest, image: ImageDataManifest, manifest: DatasetManifest) -> typing.Dict:
        raise NotImplementedError
