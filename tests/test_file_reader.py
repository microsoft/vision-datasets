import contextlib
import multiprocessing
import os
import pathlib
import pickle
import tempfile
import unittest
import zipfile

from vision_datasets.common import FileReader
from vision_datasets.common.data_reader.file_reader import MultiProcessZipFile


def open_zipfile(zip_file, filename, queue):
    queue.put(zip_file.open(filename).read())


class TestMultiProcessZipFile(unittest.TestCase):
    def test_single_process(self):
        with self._with_test_zip({'test.txt': b'contents'}) as zip_filepath:
            zip_file = MultiProcessZipFile(zip_filepath)
            with zip_file.open('test.txt') as z:
                self.assertEqual(z.read(), b'contents')
            zip_file.close()

    def test_access_from_multiple_process(self):
        with self._with_test_zip({'test.txt': b'contents'}) as zip_filepath:
            zip_file = MultiProcessZipFile(zip_filepath)
            queue = multiprocessing.Queue()
            processes = [multiprocessing.Process(target=open_zipfile, args=(zip_file, 'test.txt', queue)) for i in
                         range(5)]
            [p.start() for p in processes]
            [p.join() for p in processes]

            self.assertEqual(queue.get(False), b'contents')
            self.assertEqual(queue.get(False), b'contents')
            self.assertEqual(queue.get(False), b'contents')
            self.assertEqual(queue.get(False), b'contents')
            self.assertEqual(queue.get(False), b'contents')
            self.assertTrue(queue.empty())

    def test_pickle(self):
        with self._with_test_zip({'test.txt': b'contents'}) as zip_filepath:
            zip_file = MultiProcessZipFile(zip_filepath)
            with zip_file.open('test.txt') as z:
                self.assertEqual(z.read(), b'contents')

            serialized = pickle.dumps(zip_file)
            deserialized = pickle.loads(serialized)

            with deserialized.open('test.txt') as z:
                self.assertEqual(z.read(), b'contents')

            deserialized.close()
            zip_file.close()

    @staticmethod
    @contextlib.contextmanager
    def _with_test_zip(contents):
        """
        Args:
            contents: {filename: binary_contents} Files to be put in the test zip file.
        """

        with tempfile.TemporaryDirectory() as tempdir:
            zip_filepath = pathlib.Path(tempdir) / 'test.zip'
            with zipfile.ZipFile(zip_filepath, 'w') as f:
                for filename, bin_contents in contents.items():
                    f.writestr(filename, bin_contents)

            yield zip_filepath


class TestFileReader(unittest.TestCase):
    def test_read(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with zipfile.ZipFile(os.path.join(tempdir, 'test.zip'), 'w') as f:
                f.writestr('test.txt', b'zip_contents')
            with open(os.path.join(tempdir, 'test.txt'), 'w') as f:
                f.write('txt_contents')

            reader = FileReader()
            file = reader.open(os.path.join(tempdir, 'test.zip') + '@test.txt')
            self.assertIsNotNone(file)
            self.assertEqual(file.read(), b'zip_contents')
            file.close()

            with reader.open(os.path.join(tempdir, 'test.txt')) as f:
                self.assertEqual(f.read(), 'txt_contents')
            reader.close()


if __name__ == '__main__':
    unittest.main()
