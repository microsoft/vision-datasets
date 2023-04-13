from typing import Union, List, Dict
from urllib import parse as urlparse
from .data_manifest import DatasetManifest, ImageDataManifest
import os
import pathlib
from ..util import is_url


def generate_multitask_dataset_manifest(manifest_by_task: Dict[str, DatasetManifest]):
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


def unix_path(path: Union[pathlib.Path, str]) -> Union[pathlib.Path, str]:
    assert path is not None

    if isinstance(path, pathlib.Path):
        return path.as_posix()

    return path.replace('\\', '/')


def construct_full_path_generator(dirs: List[str]):
    """
    generate a function that appends dirs to a provided path, if dirs is empty, just return the path
    Args:
        dirs (str): dirs to be appended to a given path. None or empty str in dirs will be filtered.

    Returns:
        full_path_func: a func that appends dirs to a given path

    """
    dirs = [x for x in dirs if x]

    if dirs:
        def full_path_func(path: Union[pathlib.Path, str]):
            if isinstance(path, pathlib.Path):
                path = path.as_posix()
            to_join = [x for x in dirs + [path] if x]
            return unix_path(os.path.join(*to_join))
    else:
        full_path_func = unix_path

    return full_path_func


def construct_full_url_generator(container_sas: str):
    if not container_sas:
        return unix_path

    def add_path_to_url(url, path_or_dir):
        assert url

        if not path_or_dir:
            return url

        parts = urlparse.urlparse(url)
        path = unix_path(os.path.join(parts[2], path_or_dir))
        url = urlparse.urlunparse((parts[0], parts[1], path, parts[3], parts[4], parts[5]))

        return url

    def func(file_path):
        file_path = file_path.replace('.zip@', '/')  # cannot read from zip file with path targeting a url
        return add_path_to_url(container_sas, file_path)

    return func


def construct_full_url_or_path_generator(container_sas_or_root_dir, prefix_dir=None):
    if container_sas_or_root_dir and is_url(container_sas_or_root_dir):
        return lambda path: construct_full_url_generator(container_sas_or_root_dir)(construct_full_path_generator([prefix_dir])(path))
    else:
        return lambda path: construct_full_path_generator([container_sas_or_root_dir, prefix_dir])(path)
