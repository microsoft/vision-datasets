import os
import zipfile
from urllib import parse as urlparse
from urllib.parse import quote
from urllib.request import urlopen


def is_url(candidate: str):
    """
    necessary condition to be a url (not sufficient)
    Args:
        candidate (str):

    Returns:
        whether it could be a sas url or not

    """
    try:
        result = urlparse.urlparse(candidate)
        return result.scheme and result.netloc
    except ValueError:
        return False


class MultiProcessZipFile:
    """ZipFile which is readable from multi processes"""

    def __init__(self, filename):
        self.filename = filename
        self.zipfiles = {}

    def open(self, file):
        if os.getpid() not in self.zipfiles:
            self.zipfiles[os.getpid()] = zipfile.ZipFile(self.filename)
        return self.zipfiles[os.getpid()].open(file)

    def close(self):
        for z in self.zipfiles.values():
            z.close()
        self.zipfiles = {}

    def __getstate__(self):
        return {'filename': self.filename}

    def __setstate__(self, state):
        self.filename = state['filename']
        self.zipfiles = {}


class FileReader:
    """Reader to support <zip_filename>@<entry_name> style filename."""

    def __init__(self):
        self.zip_files = {}

    def open(self, name, mode='r'):
        # read file from url
        if is_url(name):
            return urlopen(self._encode_non_ascii(name))

        # read file from local zip: <zip_filename>@<entry_name>, e.g. images.zip@1.jpg
        if '@' in name:
            zip_path, file_path = name.split('@')
            if zip_path not in self.zip_files:
                self.zip_files[zip_path] = MultiProcessZipFile(zip_path)
            return self.zip_files[zip_path].open(file_path)

        # read file from local dir
        return open(name, mode)

    def close(self):
        for zip_file in self.zip_files.values():
            zip_file.close()
        self.zip_files = {}

    @staticmethod
    def _encode_non_ascii(s):
        return ''.join([c if ord(c) < 128 else quote(c) for c in s])
