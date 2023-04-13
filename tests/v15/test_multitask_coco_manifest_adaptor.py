import json
import pytest
import pathlib
import tempfile
import itertools
from vision_datasets.common.manifest_v15.coco_data_manifest_adaptor import ManifestAdaptorFactory, DatasetTypes
from .util import coco_dict_to_manifest, coco_database


def two_tasks_test_cases():
    tasks = list(coco_database.keys())
    two_tasks = list(itertools.product(tasks, tasks))
    coco_dicts = [list(itertools.product(coco_database[task1], coco_database[task2])) for task1, task2 in two_tasks]
    assert len(two_tasks) == len(coco_dicts)
    tasks_coco_dict = [(two_tasks[i], y) for i, x in enumerate(coco_dicts) for y in x]
    return tasks_coco_dict


@pytest.mark.parametrize("tasks, coco_dicts", two_tasks_test_cases())
def test_create_multitask_data_manifest_2_tasks(tasks, coco_dicts):
    assert len(tasks) == len(coco_dicts)
    task_names = [f'{i}_{task}' for i, task in enumerate(tasks)]
    adaptor = ManifestAdaptorFactory.create(DatasetTypes.MULTITASK, data_types={x: y for x, y in zip(task_names, tasks)})
    coco_files = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in range(len(tasks)):
            dm1_path = pathlib.Path(temp_dir) / f'coco{i}.json'
            dm1_path.write_text(json.dumps(coco_dicts[i]))
            coco_files[task_names[i]] = dm1_path
        adaptor.create_dataset_manifest(coco_files, temp_dir)
        
        # TODO: need to implement more checks
