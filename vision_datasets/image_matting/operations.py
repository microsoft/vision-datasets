from ..common import DatasetTypes, GenerateCocoDictBase, DatasetManifest, ImageDataManifest, ImageLabelManifest, SampleByNumSamples, SampleStrategyType, SingleTaskMerge, Spawn, Split, \
    CocoDictGeneratorFactory, ManifestMergeStrategyFactory, SampleStrategyFactory, SpawnFactory, SplitFactory, StandAloneImageListGeneratorFactory, GenerateStandAloneImageListBase, FileReader
from ..common.base64_utils import Base64Utils
_DATA_TYPE = DatasetTypes.IMAGE_MATTING


@CocoDictGeneratorFactory.register(_DATA_TYPE)
class ImageMattingCocoDictGenerator(GenerateCocoDictBase):
    def process_labels(self, coco_ann, label: ImageLabelManifest):
        coco_ann['label'] = label.label_path


ManifestMergeStrategyFactory.direct_register(SingleTaskMerge, _DATA_TYPE)

SampleStrategyFactory.direct_register(SampleByNumSamples, _DATA_TYPE, SampleStrategyType.NumSamples)

SpawnFactory.direct_register(Spawn, _DATA_TYPE)
SplitFactory.direct_register(Split, _DATA_TYPE)


@StandAloneImageListGeneratorFactory.register(_DATA_TYPE)
class ImageMattingStandAloneImageListGenerator(GenerateStandAloneImageListBase):
    def __init__(self, flatten: bool) -> None:
        super().__init__(flatten)

    def _generate_label(self, label: ImageLabelManifest, image: ImageDataManifest, manifest: DatasetManifest) -> dict:
        file_reader = FileReader()
        return {"matting_image": Base64Utils.file_to_b64_str(label.label_path, file_reader)}
