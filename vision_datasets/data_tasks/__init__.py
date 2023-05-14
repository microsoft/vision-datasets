from .image_caption import ImageCaptionCocoManifestAdaptor
from .image_classification import MultiClassClassificationCocoManifestAdaptor, MultiLabelClassificationCocoManifestAdaptor
from .image_matting import ImageMattingCocoManifestAdaptor
from .image_object_detection import ImageObjectDetectionCocoManifestAdaptor
from .image_regression import ImageRegressionCocoManifestAdaptor
from .image_text_matching import ImageTextMatchingCocoManifestAdaptor
from .multi_task import MultiTaskCocoManifestAdaptor
from .text_2_image_retrieval import Text2ImageRetrievalCocoManifestAdaptor

__all__ = ['ImageCaptionCocoManifestAdaptor',
           'MultiClassClassificationCocoManifestAdaptor', 'MultiLabelClassificationCocoManifestAdaptor',
           'ImageMattingCocoManifestAdaptor',
           'ImageObjectDetectionCocoManifestAdaptor',
           'ImageRegressionCocoManifestAdaptor',
           'ImageTextMatchingCocoManifestAdaptor',
           'MultiTaskCocoManifestAdaptor',
           'Text2ImageRetrievalCocoManifestAdaptor']
