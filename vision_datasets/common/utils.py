from typing import Union, List
from urllib import parse as urlparse
import os
import pathlib


def deep_merge(*dicts):
    merged = {}

    for d in dicts:
        if not isinstance(d, dict):
            continue

        for key, value in d.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value

    return merged


def can_be_url(candidate: Union[str, pathlib.Path]):
    """
    necessary conditions for candidate to be a url (not sufficient)
    Args:
        candidate (str):

    Returns:
        whether it could be a url or not

    """
    try:
        if not candidate or not isinstance(candidate, str):
            return False

        result = urlparse.urlparse(candidate)
        return result.scheme and result.netloc
    except ValueError:
        return False


def unix_path(path: Union[pathlib.Path, str]) -> Union[pathlib.Path, str]:
    if path is None:
        raise ValueError

    if isinstance(path, pathlib.Path):
        return path.as_posix()

    return path.replace('\\', '/')


def _construct_full_path_generator(dirs: List[str]):
    """
    Construct a function that appends dirs to a provided path.

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


def _construct_full_url_generator(container_url: str):
    if not container_url:
        return unix_path

    def add_path_to_url(url, path_or_dir):
        if not url:
            raise ValueError

        if not path_or_dir:
            return url

        parts = urlparse.urlparse(url)
        path = unix_path(os.path.join(parts[2], path_or_dir))
        url = urlparse.urlunparse((parts[0], parts[1], path, parts[3], parts[4], parts[5]))

        return url

    def func(file_path):
        file_path = file_path.replace('.zip@', '/')  # cannot read from zip file with path targeting a url
        return add_path_to_url(container_url, file_path)

    return func


def construct_full_url_or_path_func(url_or_root_dir: Union[str, pathlib.Path], prefix_dir: Union[str, pathlib.Path] = None):
    if url_or_root_dir and can_be_url(url_or_root_dir):
        return lambda path: _construct_full_url_generator(url_or_root_dir)(_construct_full_path_generator([prefix_dir])(path))
    else:
        return lambda path: _construct_full_path_generator([url_or_root_dir, prefix_dir])(path)
