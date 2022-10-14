# Helpful commands

- `vision_download`: help you download the dataset files to local disk for consumption, it can be downloaded/converted to TSV directly as well
- `vision_check_dataset`: check if a dataset or [coco json + images] is problematic or not.
- `vision_convert_to_tsv`: convert a dataset or [coco json + images] to TSV format, currently only classification, object detection and caption tasks are supported, TSV format doc can be found at [`TSV_FORMAT.md`](./TSV_FORMAT.md)
- `vision_convert_tsv_to_coco`: convert TSV file to [coco json + images].
- `vision_convert_od_to_ic`: convert a detection dataset to classification dataset (with or without augmentations).
- `vision_merge_datasets`: merge multiple datasets into one.

For each commoand, run `command -h` for more details.