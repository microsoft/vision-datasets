import multiprocessing
import os
import tempfile
import unittest
import zipfile

from vision_datasets.common.util import FileReader, MultiProcessZipFile


def open_zipfile(zip_file, filename, queue):
    queue.put(zip_file.open(filename).read())


class TestMultiProcessZipFile(unittest.TestCase):
    def test_single_process(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with zipfile.ZipFile(os.path.join(tempdir, 'test.zip'), 'w') as f:
                f.writestr('test.txt', b'contents')
            zip_file = MultiProcessZipFile(os.path.join(tempdir, 'test.zip'))
            with zip_file.open('test.txt') as z:
                self.assertEqual(z.read(), b'contents')
            zip_file.close()

    def test_access_from_multiple_process(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with zipfile.ZipFile(os.path.join(tempdir, 'test.zip'), 'w') as f:
                f.writestr('test.txt', b'contents')

            zip_file = MultiProcessZipFile(os.path.join(tempdir, 'test.zip'))
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
