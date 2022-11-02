import setuptools
from os import path

VERSION = '0.2.22'

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
                 python_requires='>=3.6',
                 license='MIT',
                 keywords='vision datasets classification detection',
                 packages=setuptools.find_packages(),
                 package_data={'': ['resources/*']},
                 install_requires=[
                     'numpy>=1.18.3',
                     'Pillow>=6.2.2',
                     'requests>=2.23.0',
                     'tenacity>=6.2.0',
                     'tqdm',
                     'matplotlib'
                 ],
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: 3.9',
                     'Programming Language :: Python :: 3.10',
                 ],
                 extras_require={'run': ['torch>=1.6.0', 'torchvision>=0.7.0']},
                 entry_points={
                     'console_scripts': ['vision_download=vision_datasets.commands.download_dataset:main',
                                         'vision_merge_datasets=vision_datasets.commands.merge_datasets:main',

                                         'vision_check_dataset=vision_datasets.commands.check_dataset:main',
                                         'vision_convert_od_to_ic=vision_datasets.commands.converter_od_to_ic:main',
                                         'vision_convert_local_dir_ic_data=vision_datasets.commands.convert_local_dir_ic_data:main',
                                         'vision_convert_to_tsv=vision_datasets.commands.converter_to_tsv:main',
                                         'vision_convert_tsv_to_coco=vision_datasets.commands.converter_tsv_to_coco:main',

                                         'vision_tsv_to_iris=vision_datasets.commands.converter_tsv_to_iris:main',
                                         'vision_yolo_to_iris=vision_datasets.commands.converter_yolo_darknet_to_iris:main',
                                         'vision_gen_metafile=vision_datasets.commands.generate_image_meta_info:main',
                                         'vision_convert_to_aml_coco=vision_datasets.commands.converter_to_aml_coco:main']
                 })
