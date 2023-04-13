import copy
import logging
from ...v15.common.constants import DatasetTypes
from ...common.dataset_info import MultiTaskDatasetInfo
from ...common.util import FileReader
from .utils import generate_multitask_dataset_manifest, construct_full_url_or_path_generator
from .data_manifest import DatasetManifest, ImageDataManifest


logger = logging.getLogger(__name__)


class IrisManifestAdaptor:
    """
    Adaptor for generating dataset manifest from iris format
    """

    @staticmethod
    def create_dataset_manifest(dataset_info, usage: str, container_sas_or_root_dir: str = None):
        """

        Args:
            dataset_info (MultiTaskDatasetInfo or .DatasetInfo):  dataset info
            usage (str): which usage of data to construct
            container_sas_or_root_dir (str): sas url if the data is store in a azure blob container, or a local root dir
        """
        assert dataset_info
        assert usage

        if dataset_info.type in [DatasetTypes.IMCAP, DatasetTypes.IMAGE_TEXT_MATCHING, DatasetTypes.IMAGE_MATTING, DatasetTypes.IMAGE_REGRESSION, DatasetTypes.IMAGE_RETRIEVAL]:
            raise ValueError(f'Iris format is not supported for {dataset_info.type} task, please use COCO format!')
        if isinstance(dataset_info, MultiTaskDatasetInfo):
            dataset_manifest_by_task = {k: IrisManifestAdaptor.create_dataset_manifest(task_info, usage, container_sas_or_root_dir) for k, task_info in dataset_info.sub_task_infos.items()}
            return generate_multitask_dataset_manifest(dataset_manifest_by_task)
        if usage not in dataset_info.index_files:
            return None

        file_reader = FileReader()

        dataset_info = copy.deepcopy(dataset_info)
        get_full_sas_or_path = construct_full_url_or_path_generator(container_sas_or_root_dir, dataset_info.root_folder)

        max_index = 0
        labelmap = None
        if not dataset_info.labelmap:
            logger.warning(f'{dataset_info.name}: labelmap is missing!')
        else:
            # read tag names
            with file_reader.open(get_full_sas_or_path(dataset_info.labelmap), encoding='utf-8') as file_in:
                labelmap = [IrisManifestAdaptor._purge_line(line) for line in file_in if IrisManifestAdaptor._purge_line(line) != '']

        # read image width and height
        img_wh = None
        if dataset_info.image_metadata_path:
            img_wh = IrisManifestAdaptor._load_img_width_and_height(file_reader, get_full_sas_or_path(dataset_info.image_metadata_path))

        # read image index files
        images = []
        with file_reader.open(get_full_sas_or_path(dataset_info.index_files[usage])) as file_in:
            for line in file_in:
                line = IrisManifestAdaptor._purge_line(line)
                if not line:
                    continue
                parts = line.rsplit(' ', maxsplit=1)  # assumption: only the image file path can have spaces
                img_path = parts[0]
                label_or_label_file = parts[1] if len(parts) == 2 else None

                w, h = img_wh[img_path] if img_wh else (None, None)
                if DatasetTypes.is_classification(dataset_info.type):
                    img_labels = [int(x) for x in label_or_label_file.split(',')] if label_or_label_file else []
                else:
                    img_labels = IrisManifestAdaptor._load_detection_labels_from_file(file_reader, get_full_sas_or_path(label_or_label_file)) if label_or_label_file else []

                if not labelmap and img_labels:
                    c_indices = [x[0] for x in img_labels] if isinstance(img_labels[0], list) else img_labels
                    max_index = max(max(c_indices), max_index)

                images.append(ImageDataManifest(img_path, get_full_sas_or_path(img_path), w, h, img_labels))

            if not labelmap:
                labelmap = [str(x) for x in range(max_index + 1)]
            file_reader.close()
        return DatasetManifest(images, labelmap, dataset_info.type)

    @staticmethod
    def _load_img_width_and_height(file_reader, file_path):
        img_wh = dict()
        with file_reader.open(file_path) as file_in:
            for line in file_in:
                line = IrisManifestAdaptor._purge_line(line)
                if line == '':
                    continue
                location, w, h = line.split()
                img_wh[location] = (int(w), int(h))

        return img_wh

    @staticmethod
    def _load_detection_labels_from_file(file_reader, image_label_file_path):

        with file_reader.open(image_label_file_path) as label_in:
            label_lines = [IrisManifestAdaptor._purge_line(line) for line in label_in]

        img_labels = []
        for label_line in label_lines:
            parts = label_line.split()

            assert len(parts) == 5  # regions
            box = [float(p) for p in parts]
            box[0] = int(box[0])
            img_labels.append(box)

        return img_labels

    @staticmethod
    def _purge_line(line):
        if not isinstance(line, str):
            line = line.decode('utf-8')

        return line.strip()
