import setuptools
from os import path

VERSION = '1.0.11'

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

setuptools.setup(name='vision_datasets',
                 author='Ping Jin, Shohei Ono',
                 description='A utility repo for vision dataset access and management.',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 url='https://github.com/microsoft/vision-datasets',
                 version=VERSION,
                 python_requires='>=3.8',
                 license='MIT',
                 keywords='vision datasets classification detection',
                 packages=setuptools.find_packages(),
                 package_data={'': ['resources/*']},
                 install_requires=[
                     'azure-identity',
                     'azure-storage-blob',
                     'numpy>=1.18.3',
                     'Pillow>=6.2.2',
                     'requests>=2.23.0',
                     'tenacity>=6.2.0',
                     'tqdm'
                 ],
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: 3.9',
                     'Programming Language :: Python :: 3.10',
                 ],
                 extras_require={
                     'torch': ['torch>=1.6.0'],
                     'plot': ['matplotlib'],
                 },
                 entry_points={
                     'console_scripts': ['vision_download=vision_datasets.commands.download_dataset:main',
                                         'vision_check_dataset=vision_datasets.commands.check_dataset:main',
                                         'vision_transform_images=vision_datasets.commands.transform_images:main',
                                         'vision_convert_od_to_ic=vision_datasets.commands.converter_od_to_ic:main',
                                         'vision_convert_to_aml_coco=vision_datasets.commands.converter_to_aml_coco:main',
                                         'vision_list_supported_operations=vision_datasets.commands.list_operations_by_data_type:main',
                                         'vision_convert_to_line_oriented_format=vision_datasets.commands.converter_to_line_oriented_format:main']
                 })
