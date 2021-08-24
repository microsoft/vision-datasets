import logging

from .base_dataset import BaseDataset
from .constants import DatasetTypes
from .dataset_info import BaseDatasetInfo
from .data_manifest import DatasetManifest
from .image_loader import PILImageLoader
from .util import FileReader

logger = logging.getLogger(__name__)


class ManifestDataset(BaseDataset):
    """Dataset class that accesses data from dataset manifest
    """

    def __init__(self, dataset_info: BaseDatasetInfo, dataset_manifest: DatasetManifest, coordinates='relative', dataset_resources=None):
        """

        Args:
            dataset_info (BaseDatasetInfo): dataset info, containing high level information about the dataset, such as name, type, description, etc
            dataset_manifest (DatasetManifest): dataset manifest containing meta data such as image paths, annotations, etc
            coordinates (str): 'relative' or 'absolute', indicating the desired format of the bboxes returned.
            dataset_resources (str): disposable resources associated with this dataset
        """
        assert dataset_manifest
        assert coordinates in ['relative', 'absolute']

        super().__init__(dataset_info)

        self.dataset_manifest = dataset_manifest
        self._coordinates = coordinates
        self._file_reader = FileReader()
        self.dataset_resources = dataset_resources

    @property
    def labels(self):
        return self.dataset_manifest.labelmap

    def __len__(self):
        return len(self.dataset_manifest.images)

    def _get_single_item(self, index):
        image_manifest = self.dataset_manifest.images[index]
        image = self._load_image(image_manifest.img_path)
        target = image_manifest.labels
        if self._coordinates == 'relative':
            w, h = image.size
            target = ManifestDataset._box_convert_to_relative(image_manifest.labels, w, h, self.dataset_info)

        return image, target, str(index)

    def close(self):
        self._file_reader.close()

    def _load_image(self, filepath):
        full_path = filepath.replace('\\', '/')
        try:
            with self._file_reader.open(full_path, 'rb') as f:
                return PILImageLoader.load_from_stream(f)
        except Exception:
            logger.exception(f'Failed to load an image with path: {full_path}')
            raise

    @staticmethod
    def _box_convert_to_relative(target, w, h, dataset_info):
        # Convert absolute coordinates to relative coordinates.
        # Example: for image with size (200, 200), (1, 100, 100, 200, 200) => (1, 0.5, 0.5, 1.0, 1.0)
        if dataset_info.type == DatasetTypes.MULTITASK:
            return {task_name: ManifestDataset._box_convert_to_relative(task_target, w, h, dataset_info.sub_task_infos[task_name]) for task_name, task_target in target.items()}
        if dataset_info.type == DatasetTypes.OD:
            return [[t[0], t[1] / w, t[2] / h, t[3] / w, t[4] / h] for t in target]

        return target
