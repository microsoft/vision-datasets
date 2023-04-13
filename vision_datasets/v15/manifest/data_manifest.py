import collections
import copy
import logging
import random
from PIL import Image
import numpy as np
from typing import List

from .iris_data_manifest_adaptor import IrisManifestAdaptor
from .coco_data_manifest_adaptor import ManifestAdaptorFactory
from ...v15.common.constants import DatasetTypes, AnnotationFormats
from ...common.util import FileReader

logger = logging.getLogger(__name__)


class ImageDataManifest:
    """
    Encapsulates the information and annotations of an image.

    img_path could be 1. a local path 2. a local path in a non-compressed zip file (`c:\a.zip@1.jpg`) or 3. a url.
    label_file_paths is a list of paths that have the same format with img_path
    """

    def __init__(self, id, img_path, width, height, labels, label_file_paths=None, labels_extra_info: dict = None):
        """
        Args:
            id (int or str): image id
            img_path (str): path to image
            width (int): image width
            height (int): image height
            labels (list or dict):
                classification: [c_id] for multiclass and only one c_id is allowed, [c_id1, c_id2, ...] for multilabel;
                detection: [[c_id, left, top, right, bottom], ...] (absolute coordinates);
                image_caption: [caption1, caption2, ...];
                image_text_matching: [(text1, match (0 or 1), text2, match (0 or 1), ...)];
                multitask: dict[task, labels];
                image_matting: [mask1, mask2, ...], each mask is a 2D numpy array that has the same width and height with the image;
                image_regression: [target], only one target is allowed;
                image_retrieval: [query1, query2, ...]
            label_file_paths (list): list of paths of the image label files. "label_file_paths" only works for image matting task.
            labels_extra_info (dict[string, list]]): extra information about this image's labels
                Examples: 'iscrowd'
        """

        self.id = id
        self.img_path = img_path
        self.width = width
        self.height = height
        self._labels = labels
        self.label_file_paths = label_file_paths
        self.labels_extra_info = labels_extra_info or {}

    @property
    def labels(self):
        if self._labels:
            return self._labels
        elif self.label_file_paths:  # lazy load only for image matting
            file_reader = FileReader()
            self._labels = []
            for label_file_path in self.label_file_paths:
                with file_reader.open(label_file_path) as f:
                    label = np.asarray(Image.open(f))
                    self._labels.append(label)
            file_reader.close()
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value


class CategoryManifest:
    def __init__(self, id, name: str, super_category: str):
        self.id = id
        self.name = name
        self.super_category = super_category


class DatasetManifest:
    """
    Encapsulates every information about a dataset including categories, images (width, height, path to image), and annotations. Information about each image is encapsulated in ImageDataManifest.
    """

    def __init__(self, images: List[ImageDataManifest], categories: List[CategoryManifest], data_type):
        """

        Args:
            images (list): image manifest
            categories (list or dict): labels or labels by task name
            data_type (str or dict) : data type, or data type by task name
        """

        assert data_type and data_type != DatasetTypes.MULTITASK, 'For multitask, data_type should be a dict mapping task name to concrete data type.'

        if isinstance(categories, dict):
            assert isinstance(data_type, dict), 'categories being a dict indicating this is a multitask dataset, however the data_type is not a dict.'
            assert categories.keys() == data_type.keys(), f'mismatched task names in categories and task_type: {categories.keys()} vs {data_type.keys()}'

        self.images = images
        self.categories = categories
        self.data_type = data_type

        self._task_names = sorted(categories.keys()) if self.is_multitask else None

    @staticmethod
    def create_dataset_manifest(dataset_info, usage: str, container_sas_or_root_dir: str = None):
        annotation_format = AnnotationFormats[dataset_info.data_format]

        if annotation_format == AnnotationFormats.IRIS:
            return IrisManifestAdaptor.create_dataset_manifest(dataset_info, usage, container_sas_or_root_dir)

        if annotation_format == AnnotationFormats.COCO:
            from .utils import construct_full_url_or_path_generator
            container_sas_or_root_dir = construct_full_url_or_path_generator(container_sas_or_root_dir, dataset_info.root_folder)('')
            if dataset_info.type == DatasetTypes.MULTITASK:
                coco_file_by_task = {k: sub_taskinfo.index_files.get(usage) for k, sub_taskinfo in dataset_info.sub_task_infos.items()}
                data_type_by_task = {k: sub_taskinfo.type for k, sub_taskinfo in dataset_info.sub_task_infos.items()}
                adaptor = ManifestAdaptorFactory.create(DatasetTypes.MULTITASK, data_type_by_task)
                return adaptor.create_dataset_manifest(coco_file_by_task, container_sas_or_root_dir)

            adaptor = ManifestAdaptorFactory.create(dataset_info.type, data_type_by_task)
            return adaptor.create_dataset_manifest(dataset_info.index_files.get(usage), container_sas_or_root_dir)

    @property
    def is_multitask(self):
        """
        is this dataset multi-task dataset or not
        """

        return isinstance(self.data_type, dict)

    def __len__(self):
        return len(self.images)

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

    def _get_cid(self, category, task_name=None):
        if task_name:  # multitask
            cnt = 0
            for t_name in self._task_names:
                if t_name == task_name:
                    break
                cnt += len(self.categories[t_name])

            return cnt + self._get_cid(category)
        elif isinstance(category, int):  # classification
            return category
        elif isinstance(category, list):  # detection
            return category[0]
        else:
            raise RuntimeError(f'unknown type of label: {type(category)}')

    def _is_negative(self, labels):
        n_labels = len(labels) if not self.is_multitask else sum([len(x) for x in labels.values()])
        return n_labels == 0

    def generate_coco_annotations(self):
        """
        Generate coco annotations, working for single task classification, detection, caption, and image regression only

        Returns:
            A dict of annotation data ready for coco json dump

        """

        images = []
        for i, x in enumerate(self.images):
            image = {'id': i + 1, 'file_name': x.img_path}
            if x.width:
                image['width'] = x.width
            if x.height:
                image['height'] = x.height
            images.append(image)

        annotations = []
        for img_id, img in enumerate(self.images):
            for ann in img.labels:
                coco_ann = {
                    'id': len(annotations) + 1,
                    'image_id': img_id + 1,
                }

                if DatasetTypes.is_classification(self.data_type):
                    coco_ann['category_id'] = ann + 1
                elif self.data_type == DatasetTypes.OD:
                    coco_ann['category_id'] = ann[0] + 1
                    coco_ann['bbox'] = [ann[1], ann[2], ann[3] - ann[1], ann[4] - ann[2]]
                elif self.data_type == DatasetTypes.IMCAP:
                    coco_ann['caption'] = ann
                elif self.data_type == DatasetTypes.IMAGE_REGRESSION:
                    coco_ann['target'] = ann
                elif self.data_type == DatasetTypes.IMAGE_RETRIEVAL:
                    if isinstance(ann, str):
                        coco_ann['query'] = ann
                    else:
                        coco_ann['category_id'] = ann[0] + 1
                        coco_ann['query'] = ann[1]
                else:
                    raise ValueError(f'Unsupported data type {self.data_type}')

                annotations.append(coco_ann)

        coco_dict = {'images': images, 'annotations': annotations}
        if self.data_type not in [DatasetTypes.IMCAP, DatasetTypes.IMAGE_REGRESSION, DatasetTypes.IMAGE_RETRIEVAL]:
            coco_dict['categories'] = [{'id': i + 1, 'name': x.name} for i, x in enumerate(self.categories)]

        if self.data_type == DatasetTypes.IMAGE_RETRIEVAL and self.categories:
            coco_dict['categories'] = [{'id': i + 1, 'name': x[0], 'supercategory': x[1]} for i, x in enumerate(self.categories)]

        return coco_dict

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
            return DatasetManifest([], self.categories, self.data_type), DatasetManifest(self.images, self.categories, self.data_type)

        if int(len(self.images) * train_ratio) == len(self.images):
            return DatasetManifest(self.images, self.categories, self.data_type), DatasetManifest([], self.categories, self.data_type)

        rng = random.Random(random_seed)
        images = list(self.images)
        rng.shuffle(images)

        train_imgs = []
        val_imgs = []
        n_train_imgs_by_class = [0] * len(self.categories) if not self.is_multitask else [0] * sum([len(x) for x in self.categories.values()])
        n_val_imgs_by_class = [0] * len(self.categories) if not self.is_multitask else [0] * sum([len(x) for x in self.categories.values()])
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

        return DatasetManifest(train_imgs, self.categories, self.data_type), DatasetManifest(val_imgs, self.categories, self.data_type)

    def sample_categories(self, category_indices: List):
        """
        Sample a new dataset of selected categories. Works for single IC and OD dataset only.
        Args:
            category_indices: indices of the selected categories

        Returns:
            a sampled dataset with selected categories

        """

        assert self.data_type in [DatasetTypes.IC_MULTICLASS, DatasetTypes.IC_MULTILABEL, DatasetTypes.OD]
        assert category_indices
        assert max(category_indices) < len(self.categories)

        category_id_remap = {o_cid: n_cid for n_cid, o_cid in enumerate(category_indices)}
        new_categories = [self.categories[x] for x in category_indices]
        new_images = []
        for img in self.images:
            new_img = copy.deepcopy(img)
            if DatasetTypes.is_classification(self.data_type):
                new_img.labels = [category_id_remap[x] for x in new_img.labels if x in category_id_remap]
            else:
                new_img.labels = [[category_id_remap[x[0]], x[1], x[2], x[3], x[4]] for x in img.labels if x[0] in category_id_remap]

            new_images.append(new_img)
        return DatasetManifest(new_images, new_categories, self.data_type)

    def sample_subset(self, num_samples, with_replacement=False, random_seed=0):
        """
        Sample a subset of num_samples images. When with_replacement is False and num_samples is larger than the dataset, the whole dataset will be returned
        Args:
            num_samples (int): number of images to be sampled
            with_replacement (bool): with replacement or not
            random_seed (int): random seed

        Returns:
            a sampled dataset
        """

        rnd = random.Random(random_seed)
        if not with_replacement:
            if num_samples >= len(self.images):
                sampled_images = self.images
            else:
                sampled_images = rnd.sample(self.images, num_samples)
        else:
            sampled_images = [rnd.choice(self.images) for _ in range(num_samples)]

        sampled_images = [copy.deepcopy(x) for x in sampled_images]
        return DatasetManifest(sampled_images, self.categories, self.data_type)

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
        n_imgs_by_class = [0] * len(self.categories) if not self.is_multitask else [0] * sum([len(x) for x in self.categories.values()])
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

        sampled_images = [copy.deepcopy(x) for x in sampled_images]
        return DatasetManifest(sampled_images, self.categories, self.data_type)

    def sample_subset_by_ratio(self, sampling_ratio):
        """
        Sample a dataset so that each labels appears by at least the given sampling_ratio. In case of multiclass dataset, the number of sampled images will be N * sampling_ratio.
        For multilabel or object detection datasets, the total number of images will be bigger than that.

        Args:
            sampling_ratio (float): sampling ratio. must be 0 < x < 1.

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

        sampled_images = [copy.deepcopy(self.images[i]) for i in sampled_image_ids]
        return DatasetManifest(sampled_images, self.categories, self.data_type)

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

        num_classes = len(self.categories) if not self.is_multitask else sum(len(x) for x in self.categories.values())
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

        sampled_images = [copy.deepcopy(x) for x in sampled_images]
        return DatasetManifest(sampled_images, self.categories, self.data_type)

    def remove_images_without_labels(self):
        """
        Remove images without labels.
        """
        images = [copy.deepcopy(image) for image in self.images if not self._is_negative(image.labels)]
        return DatasetManifest(images, self.categories, self.data_type)

    def spawn(self, num_samples, random_seed=0, instance_weights: List = None):
        """Spawn manifest to a size.
        To ensure each class has samples after spawn, we first keep a copy of original data, then merge with sampled data.
        If instance_weights is not provided, spawn follows class distribution.
        Otherwise spawn the dataset so that the instances follow the given weights. In this case the spawned size is not guranteed to be num_samples.

        Args:
            num_samples (int): size of spawned manifest. Should be larger than the current size.
            random_seed (int): Random seed to use.
            instance_weights (list): weight of each instance to spawn, >= 0.

        Returns:
            Spawned dataset (DatasetManifest)
        """
        assert num_samples > len(self)
        if instance_weights is not None:
            assert len(instance_weights) == len(self)
            assert all([x >= 0 for x in instance_weights])

            sum_weights = sum(instance_weights)
            # Distribute the number of num_samples to each image by the weights. The original image is subtracted.
            n_copies_per_sample = [max(0, round(w / sum_weights * num_samples - 1)) for w in instance_weights]
            spawned_images = []
            for image, n_copies in zip(self.images, n_copies_per_sample):
                spawned_images += [copy.deepcopy(image) for _ in range(n_copies)]

            sampled_manifest = DatasetManifest(spawned_images, self.categories, self.data_type)
        else:
            sampled_manifest = self.sample_subset(num_samples - len(self), with_replacement=True, random_seed=random_seed)

        # Merge with the copy of the original dataset to ensure each class has sample.
        return DatasetManifest.merge(self, sampled_manifest, flavor=0)

    @staticmethod
    def merge(*args, flavor: int = 0):
        """
        merge multiple data manifests into one.

        Args:
            args: manifests to be merged
            flavor: flavor of dataset merge (not difference for captioning)
                0: merge manifests of the same type and the same categories (for multitask, it should be same set of tasks and same categories for each task)
                1: concat manifests of the same type, the new categories are concats of all categories in all manifest (for multitask, duplicate task names are not allowed)
        """

        assert len(args) >= 1, 'less than one manifests provided, not possible to merged.'
        assert all([arg is not None for arg in args]), '"None" manifest found'

        args = [arg for arg in args if arg]
        if len(args) == 1:
            logger.warning('Only one manifest provided. Nothing to be merged.')
            return args[0]

        if any([isinstance(x.data_type, dict) for x in args]):
            assert all([isinstance(x.data_type, dict) for x in args]), 'Cannot merge multitask manifest and single task manifest'
        else:
            assert len(set([x.data_type for x in args])) == 1, 'All manifests must be of the same data type'

        if flavor == 0:
            return DatasetManifest._merge_with_same_categories(*args)
        elif flavor == 1:
            return DatasetManifest._merge_with_concat(*args)
        else:
            raise ValueError(f'Unknown flavor {flavor}.')

    @staticmethod
    def _merge_with_same_categories(*args):
        for i in range(len(args)):
            if i > 0 and args[i].categories != args[i - 1].categories:
                raise ValueError('categories must be the same for all manifests.')
            if i > 0 and args[i].data_type != args[i - 1].data_type:
                raise ValueError('Data type must be the same for all manifests.')

        images = [y for x in args for y in x.images]

        return DatasetManifest(images, args[0].categories, args[0].data_type)

    @staticmethod
    def _merge_with_concat(*args):
        data_type = args[0].data_type

        if data_type in [DatasetTypes.IMCAP, DatasetTypes.IMAGE_REGRESSION]:
            return DatasetManifest._merge_with_same_categories(args)

        if isinstance(data_type, dict):  # multitask
            categories = {}
            data_types = {}
            for manifest in args:
                for k, v in manifest.categories.items():
                    if k in categories:
                        raise ValueError(f'Failed to merge dataset manifests, as due to task with name {k} exists in more than one manifest.')

                    categories[k] = v

                for k, v in manifest.data_type.items():
                    data_types[k] = v

            return DatasetManifest([y for x in args for y in x.images], categories, data_types)

        categories = []
        images = []

        for manifest in args:
            label_offset = len(categories)
            for img_manifest in manifest.images:
                new_img_manifest = copy.deepcopy(img_manifest)
                if DatasetTypes.is_classification(data_type):
                    new_img_manifest.labels = [x + label_offset for x in new_img_manifest.labels]
                elif data_type == DatasetTypes.OD:
                    for label in new_img_manifest.labels:
                        label[0] += label_offset
                else:
                    raise ValueError(f'Unsupported type in merging {data_type}')

                images.append(new_img_manifest)
            categories.extend(manifest.categories)

        return DatasetManifest(images, categories, data_type)

    @staticmethod
    def create_multitask_manifest(manifest_by_task: dict):
        """
        Merge several manifests into a multitask dataset in a naive way, assuming images from different manifests are independent different images.
        Args:
            manifest_by_task (dict): manifest by task name

        Returns:
            a merged multitask manifest
        """

        task_names = sorted(list(manifest_by_task.keys()))
        images = []
        for task_name in task_names:
            for img in manifest_by_task[task_name].images:
                new_img = copy.deepcopy(img)
                new_img.labels = {task_name: new_img.labels}
                images.append(new_img)

        categories = {task_name: manifest_by_task[task_name].categories for task_name in task_names}
        data_types = {task_name: manifest_by_task[task_name].data_type for task_name in task_names}

        return DatasetManifest(images, categories, data_types)
