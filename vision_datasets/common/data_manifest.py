import collections
import copy
import json
import logging
import os
import random
from typing import List, Dict
from urllib import parse as urlparse

from .constants import DatasetTypes, Formats, BBoxFormat
from .dataset_info import MultiTaskDatasetInfo
from .util import is_url, FileReader

logger = logging.getLogger(__name__)


def purge_line(line):
    if not isinstance(line, str):
        line = line.decode('utf-8')

    return line.strip()


def _purge_path(path):
    assert path is not None

    return path.replace('\\', '/')


def _construct_full_path_generator(dirs: List[str]):
    """
    generate a function that appends dirs to a provided path, if dirs is empty, just return the path
    Args:
        dirs (str): dirs to be appended to a given path. None or empty str in dirs will be filtered.

    Returns:
        full_path_func: a func that appends dirs to a given path

    """
    dirs = [x for x in dirs if x]

    if dirs:
        def full_path_func(path):
            to_join = [x for x in dirs + [path] if x]
            return _purge_path(os.path.join(*to_join))
    else:
        full_path_func = _purge_path

    return full_path_func


def _add_path_to_sas(sas, path_or_dir):
    assert sas
    if not path_or_dir:
        return sas

    parts = urlparse.urlparse(sas)
    path = _purge_path(os.path.join(parts[2], path_or_dir))
    path = path.replace('.zip@', '/')  # cannot read from zip file with path targeting a url
    url = urlparse.urlunparse((parts[0], parts[1], path, parts[3], parts[4], parts[5]))
    return url


def _construct_full_sas_generator(container_sas: str):
    if not container_sas:
        return _purge_path

    def func(file_path):
        return _add_path_to_sas(container_sas, file_path)

    return func


def _construct_full_sas_or_path_generator(container_sas_or_root_dir, prefix_dir=None):
    if container_sas_or_root_dir and is_url(container_sas_or_root_dir):
        return lambda path: _construct_full_sas_generator(container_sas_or_root_dir)(_construct_full_path_generator([prefix_dir])(path))
    else:
        return lambda path: _construct_full_path_generator([container_sas_or_root_dir, prefix_dir])(path)


class ImageDataManifest:
    """
    Encapsulates the information and annotations of an image.

    img_path could be 1. a local path 2. a local path in a non-compressed zip file (`c:\a.zip@1.jpg`) or 3. a url.
    """

    def __init__(self, id, img_path, width, height, labels):
        """
        Args:
            id (int or str): image id
            img_path (str): path to image
            width (int): image width
            height (int): image height
            labels (list or dict): classification: [c_id] for multiclass, [c_id1, c_id2, ...] for multilabel; detection: [c_id, left, top, right, bottom]; dict[task, labels] for multitask dataset
        """
        self.id = id
        self.img_path = img_path
        self.width = width
        self.height = height
        self.labels = labels


class DatasetManifest:
    """
    Encapsulates every information about a dataset including labelmap, images (width, height, path to image), and annotations. Information about each image is encapsulated in ImageDataManifest.
    """

    def __init__(self, images: List[ImageDataManifest], labelmap, data_type):
        """

        Args:
            images (list): image manifest
            labelmap (list or dict): labels, or labels by task name
            data_type (str or dict) : data type, or data type by task name

        """
        assert data_type != DatasetTypes.MULTITASK, 'For multitask, data_type should be a dict mapping task name to concrete data type.'

        if isinstance(labelmap, dict):
            assert isinstance(data_type, dict), 'labelmap being a dict indicating this is a multitask dataset, however the data_type is not a dict.'
            assert labelmap.keys() == data_type.keys(), f'mismatched task names in labelmap and task_type: {labelmap.keys()} vs {data_type.keys()}'

        self.images = images
        self.labelmap = labelmap
        self.data_type = data_type

        self._task_names = sorted(labelmap.keys()) if self.is_multitask else None

    @staticmethod
    def create_dataset_manifest(dataset_info, usage: str, container_sas_or_root_dir: str = None):

        if dataset_info.data_format == Formats.IRIS:
            return IrisManifestAdaptor.create_dataset_manifest(dataset_info, usage, container_sas_or_root_dir)
        if dataset_info.data_format == Formats.COCO:
            get_full_sas_or_path = _construct_full_sas_or_path_generator(container_sas_or_root_dir, dataset_info.root_folder)('')
            if dataset_info.type == DatasetTypes.MULTITASK:
                coco_file_by_task = {k: sub_taskinfo.index_files.get(usage) for k, sub_taskinfo in dataset_info.sub_task_infos.items()}
                data_type_by_task = {k: sub_taskinfo.type for k, sub_taskinfo in dataset_info.sub_task_infos.items()}
                return CocoManifestAdaptor.create_dataset_manifest(coco_file_by_task, data_type_by_task, get_full_sas_or_path)

            return CocoManifestAdaptor.create_dataset_manifest(dataset_info.index_files.get(usage), dataset_info.type, get_full_sas_or_path)

        raise RuntimeError(f'{dataset_info.data_format} not supported yet.')

    @staticmethod
    def merge(manifest_a, manifest_b):
        assert manifest_a
        assert manifest_b

        assert manifest_a.data_type == manifest_b.data_type
        assert manifest_a.labelmap == manifest_b.labelmap

        return DatasetManifest(manifest_a.images + manifest_b.images, manifest_a.labelmap, manifest_a.data_type)

    @property
    def is_multitask(self):
        return isinstance(self.data_type, dict)

    def _add_label_count(self, labels, n_images_by_class: list):
        if self.is_multitask:
            for task_name, task_labels in labels.items():
                for label in task_labels:
                    n_images_by_class[self._get_cid(label, task_name)] += 1
        else:
            for label in labels:
                n_images_by_class[self._get_cid(label)] += 1

    def _get_label_count(self, labels, n_images_by_class: list):
        if self.is_multitask:
            return [n_images_by_class[self._get_cid(label, task_name)] for task_name, task_labels in labels.items() for label in task_labels]
        else:
            return [n_images_by_class[self._get_cid(label)] for label in labels]

    def _get_cid(self, label, task_name=None):
        if task_name:  # multitask
            cnt = 0
            for t_name in self._task_names:
                if t_name == task_name:
                    break
                cnt += len(self.labelmap[t_name])

            return cnt + self._get_cid(label)
        elif isinstance(label, int):  # classification
            return label
        elif isinstance(label, list):  # detection
            return label[0]
        else:
            raise RuntimeError(f'unknown type of label: {type(label)}')

    def _is_negative(self, labels):
        n_labels = len(labels) if not self.is_multitask else sum([len(x) for x in labels.values()])
        return n_labels == 0

    def train_val_split(self, train_ratio, random_seed=0):
        """
        Split the dataset into train and val set, with train set ratio being train_ratio.
        For multiclass dataset, the split ratio will be close to provided train_ratio, while for multilabel dataset, it is not guaranteed
        Multitask dataset and detection dataset are treated the same with multilabel dataset.
        Args:
            train_ratio(float): rough train set ratio, from 0 to 1
            random_seed: random seed

        Returns:
            train_manifest, val_manifest
        """
        if int(len(self.images) * train_ratio) == 0:
            return DatasetManifest([], self.labelmap, self.data_type), DatasetManifest(self.images, self.labelmap, self.data_type)

        if int(len(self.images) * train_ratio) == len(self.images):
            return DatasetManifest(self.images, self.labelmap, self.data_type), DatasetManifest([], self.labelmap, self.data_type)

        rng = random.Random(random_seed)
        images = list(self.images)
        rng.shuffle(images)

        train_imgs = []
        val_imgs = []
        n_train_imgs_by_class = [0] * len(self.labelmap) if not self.is_multitask else [0] * sum([len(x) for x in self.labelmap.values()])
        n_val_imgs_by_class = [0] * len(self.labelmap) if not self.is_multitask else [0] * sum([len(x) for x in self.labelmap.values()])
        test_train_ratio = (1 - train_ratio) / train_ratio
        n_train_neg = 0
        n_val_neg = 0
        for image in images:
            if self._is_negative(image.labels):
                if n_train_neg == 0 or n_val_neg / n_train_neg >= test_train_ratio:
                    n_train_neg += 1
                    train_imgs.append(image)
                else:
                    n_val_neg += 1
                    val_imgs.append(image)

                continue

            train_cnt = self._get_label_count(image.labels, n_train_imgs_by_class)
            val_cnt = self._get_label_count(image.labels, n_val_imgs_by_class)
            train_cnt_sum = sum(train_cnt) * test_train_ratio
            train_cnt_min = min(train_cnt) * test_train_ratio
            val_cnt_sum = sum(val_cnt)
            val_cnt_min = min(val_cnt)
            if val_cnt_min < train_cnt_min or (val_cnt_min == train_cnt_min and val_cnt_sum < train_cnt_sum):
                val_imgs.append(image)
                self._add_label_count(image.labels, n_val_imgs_by_class)
            else:
                train_imgs.append(image)
                self._add_label_count(image.labels, n_train_imgs_by_class)

        return DatasetManifest(train_imgs, self.labelmap, self.data_type), DatasetManifest(val_imgs, self.labelmap, self.data_type)

    def sample_few_shot_subset(self, num_samples_per_class, random_seed=0):
        """
        Sample a few-shot dataset, with the number of images per class below num_samples_per_class.
        For multiclass dataset, this is always possible, while for multilabel dataset, it is not guaranteed
        Multitask dataset and detection dataset are treated the same with multilabel dataset.

        This method tries to get balanced results.

        Note that negative images will be added to the subset up to num_samples_per_class.

        Args:
            num_samples_per_class: rough number samples per class to sample
            random_seed: random seed

        Returns:
            a sampled few-shot subset
        """

        assert num_samples_per_class > 0

        sampled_images = []
        rng = random.Random(random_seed)
        images = list(self.images)
        rng.shuffle(images)
        n_imgs_by_class = [0] * len(self.labelmap) if not self.is_multitask else [0] * sum([len(x) for x in self.labelmap.values()])
        neg_img_cnt = 0
        for image in images:
            if self._is_negative(image.labels):
                if neg_img_cnt < num_samples_per_class:
                    neg_img_cnt += 1
                    sampled_images.append(image)
                continue

            img_label_cnt = self._get_label_count(image.labels, n_imgs_by_class)

            if min(img_label_cnt) >= num_samples_per_class:
                continue

            if min(img_label_cnt) <= num_samples_per_class / 2 or max(img_label_cnt) <= 1.5 * num_samples_per_class:
                sampled_images.append(image)
                self._add_label_count(image.labels, n_imgs_by_class)

            if min(n_imgs_by_class) >= num_samples_per_class:
                break

        return DatasetManifest(sampled_images, self.labelmap, self.data_type)

    def sample_subset_by_ratio(self, sampling_ratio):
        """
        Sample a dataset so that each labels appears by at least the given sampling_ratio. In case of multiclass dataset, the number of sampled images will be N * sampling_ratio.
        For multilabel or object detection datasets, the total number of images will be bigger than that.

        Args:
            sampling_ratio (float): sampling raito. must be 0 < x < 1.

        Returns:
            A sampled dataset (DatasetManifest)
        """
        assert 0 < sampling_ratio < 1

        if self.is_multitask:
            labels = [[self._get_cid(c, t) for t, t_labels in image.labels.items() for c in t_labels] for image in self.images]
        else:
            labels = [[self._get_cid(c) for c in image.labels] for image in self.images]

        # Create a dict {label_id: [image_id, ...], ...}
        # Note that image_id can be included multiple times if the dataset is multilabel, objectdetection, or multitask.
        label_image_map = collections.defaultdict(list)
        for i, image_labels in enumerate(labels):
            if not image_labels:
                label_image_map[-1].append(i)
            for label in image_labels:
                label_image_map[label].append(i)

        # From each lists, sample max(1, N * ratio) images.
        sampled_image_ids = set()
        for image_ids in label_image_map.values():
            sampled_image_ids |= set(random.sample(image_ids, max(1, int(len(image_ids) * sampling_ratio))))

        sampled_images = [self.images[i] for i in sampled_image_ids]
        return DatasetManifest(sampled_images, self.labelmap, self.data_type)

    def sample_few_shots_subset_greedy(self, num_min_samples_per_class, random_seed=0):
        """Greedy few-shots sampling method.
        Randomly pick images from the original datasets until all classes have at least {num_min_images_per_class} tags/boxes.

        Note that images without any tag/box will be ignored. All images in the subset will have at least one tag/box.

        Args:
            num_min_samples_per_class (int): The minimum number of samples per class.
            random_seed (int): Random seed to use.

        Returns:
            A samped dataset (DatasetManifest)

        Raises:
            RuntimeError if it couldn't find num_min_samples_per_class samples for all classes
        """
        assert num_min_samples_per_class > 0
        images = list(self.images)
        rng = random.Random(random_seed)
        rng.shuffle(images)

        num_classes = len(self.labelmap) if not self.is_multitask else sum(len(x) for x in self.labelmap.values())
        total_counter = collections.Counter({i: num_min_samples_per_class for i in range(num_classes)})
        sampled_images = []
        for image in images:
            counts = collections.Counter([self._get_cid(c) for c in image.labels] if not self.is_multitask else [self._get_cid(c, t) for t, t_labels in image.labels.items() for c in t_labels])
            if set((+total_counter).keys()) & set(counts.keys()):
                total_counter -= counts
                sampled_images.append(image)

            if not +total_counter:
                break

        if +total_counter:
            raise RuntimeError(f"Couldn't find {num_min_samples_per_class} samples for some classes: {+total_counter}")

        return DatasetManifest(sampled_images, self.labelmap, self.data_type)


def _generate_multitask_dataset_manifest(manifest_by_task: Dict[str, DatasetManifest]):
    images_by_id = {}
    for task_name, task_manifest in manifest_by_task.items():
        if not task_manifest:
            continue
        for image in task_manifest.images:
            if image.id not in images_by_id:
                multi_task_image_manifest = ImageDataManifest(image.id, image.img_path, image.width, image.height, {task_name: image.labels})
                images_by_id[image.id] = multi_task_image_manifest
            else:
                images_by_id[image.id].labels[task_name] = image.labels

    if not images_by_id:
        return None

    labelmap_by_task = {k: manifest.labelmap for k, manifest in manifest_by_task.items()}
    dataset_types_by_task = {k: manifest.data_type for k, manifest in manifest_by_task.items()}
    return DatasetManifest([v for v in images_by_id.values()], labelmap_by_task, dataset_types_by_task)


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

        if isinstance(dataset_info, MultiTaskDatasetInfo):
            dataset_manifest_by_task = {k: IrisManifestAdaptor.create_dataset_manifest(task_info, usage, container_sas_or_root_dir) for k, task_info in dataset_info.sub_task_infos.items()}
            return _generate_multitask_dataset_manifest(dataset_manifest_by_task)

        if usage not in dataset_info.index_files:
            return None

        file_reader = FileReader()

        dataset_info = copy.deepcopy(dataset_info)
        get_full_sas_or_path = _construct_full_sas_or_path_generator(container_sas_or_root_dir, dataset_info.root_folder)

        max_index = 0
        labelmap = None
        if not dataset_info.labelmap:
            logger.warning(f'{dataset_info.name}: labelmap is missing!')
        else:
            # read tag names
            with file_reader.open(get_full_sas_or_path(dataset_info.labelmap)) as file_in:
                labelmap = [purge_line(line) for line in file_in if purge_line(line) != '']

        # read image width and height
        img_wh = None
        if dataset_info.image_metadata_path:
            img_wh = IrisManifestAdaptor._load_img_width_and_height(file_reader, get_full_sas_or_path(dataset_info.image_metadata_path))

        # read image index files
        images = []
        with file_reader.open(get_full_sas_or_path(dataset_info.index_files[usage])) as file_in:
            for line in file_in:
                line = purge_line(line)
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
                line = purge_line(line)
                if line == '':
                    continue
                location, w, h = line.split()
                img_wh[location] = (int(w), int(h))

        return img_wh

    @staticmethod
    def _load_detection_labels_from_file(file_reader, image_label_file_path):

        with file_reader.open(image_label_file_path) as label_in:
            label_lines = [purge_line(line) for line in label_in]

        img_labels = []
        for label_line in label_lines:
            parts = label_line.split()

            assert len(parts) == 5  # regions
            box = [float(p) for p in parts]
            box[0] = int(box[0])
            img_labels.append(box)

        return img_labels


class CocoManifestAdaptor:
    """
    Adaptor for generating manifest from coco format

    image paths should be stored under 'file_name'
    """

    @staticmethod
    def create_dataset_manifest(coco_file_path_or_url, data_type, container_sas_or_root_dir: str = None):
        """ construct a dataset manifest out of coco file
        Args:
            coco_file_path_or_url (str or dict): path or url to coco file. dict if multitask
            data_type (str or dict): type of dataset. dict if multitask
            container_sas_or_root_dir (str): container sas if resources are store in blob container, or a local dir
        """
        if not coco_file_path_or_url:
            return None

        assert data_type

        if isinstance(coco_file_path_or_url, dict):
            assert isinstance(data_type, dict)
            dataset_manifest_by_task = {k: CocoManifestAdaptor.create_dataset_manifest(coco_file_path_or_url[k], data_type[k], container_sas_or_root_dir)
                                        for k in coco_file_path_or_url}

            return _generate_multitask_dataset_manifest(dataset_manifest_by_task)

        get_full_sas_or_path = _construct_full_sas_or_path_generator(container_sas_or_root_dir)

        file_reader = FileReader()
        # read image index files
        with file_reader.open(coco_file_path_or_url if is_url(coco_file_path_or_url) else get_full_sas_or_path(coco_file_path_or_url)) as file_in:
            coco_manifest = json.load(file_in)

        file_reader.close()

        images_by_id = {img['id']: ImageDataManifest(img['id'], get_full_sas_or_path(img['file_name']), img['width'], img['height'], []) for img in coco_manifest['images']}

        label_dict_by_id = {cate['id']: cate['name'] for cate in coco_manifest['categories']}
        label_starting_idx = min(label_dict_by_id.keys())
        labelmap = [label_dict_by_id[i + label_starting_idx] for i in range(len(label_dict_by_id))]

        bbox_format = coco_manifest.get('bbox_format', BBoxFormat.LTWH)
        BBoxFormat.validate(bbox_format)

        for annotation in coco_manifest['annotations']:
            c_id = annotation['category_id'] - label_starting_idx
            if 'bbox' in annotation:
                bbox = annotation['bbox']
                if bbox_format == BBoxFormat.LTWH:
                    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                label = [c_id] + bbox
            else:
                label = c_id
            images_by_id[annotation['image_id']].labels.append(label)
        images = [x for x in images_by_id.values()]
        images.sort(key=lambda x: x.id)

        return DatasetManifest(images, labelmap, data_type)
