from typing import Dict

from .data_manifest import DatasetManifest, ImageDataManifest


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

    categories_by_task = {k: manifest.categories for k, manifest in manifest_by_task.items()}
    dataset_types_by_task = {k: manifest.data_type for k, manifest in manifest_by_task.items()}
    additional_info_by_task = {k: manifest.additional_info for k, manifest in manifest_by_task.items() if manifest.additional_info}

    return DatasetManifest([v for v in images_by_id.values()], categories_by_task, dataset_types_by_task, addtional_info=additional_info_by_task)
