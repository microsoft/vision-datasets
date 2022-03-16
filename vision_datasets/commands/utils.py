import base64
import io
import logging
import os

from vision_datasets.common.image_loader import PILImageLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TSV_FORMAT_LTRB = 'ltrb'
TSV_FORMAT_LTWH_NORM = 'ltwh-normalized'


def zip_folder(folder_name):
    import zipfile

    logger.info(f'zipping {folder_name}.')

    zip_file = zipfile.ZipFile(f'{folder_name}.zip', 'w', zipfile.ZIP_STORED)
    i = 0
    for root, dirs, files in os.walk(folder_name):
        for file in files:
            if i % 5000 == 0:
                logger.info(f'zipped {i} images')

            zip_file.write(os.path.join(root, file))
            i += 1

    zip_file.close()


def decode64_to_pil(img_b64_str):
    assert img_b64_str

    return PILImageLoader.load_from_stream(io.BytesIO(base64.b64decode(img_b64_str)))


def guess_encoding(csv_file):
    """guess the encoding of the given file https://stackoverflow.com/a/33981557/
    """
    import io
    import locale

    with io.open(csv_file, 'rb') as f:
        data = f.read(5)
    if data.startswith(b'\xEF\xBB\xBF'):  # UTF-8 with a "BOM"
        return 'utf-8-sig'
    elif data.startswith(b'\xFF\xFE') or data.startswith(b"\xFE\xFF"):
        return 'utf-16'
    else:  # in Windows, guessing utf-8 doesn't work, so we have to try
        # noinspection PyBroadException
        try:
            with io.open(csv_file, encoding='utf-8') as f:
                f.read(222222)
                return 'utf-8'
        except Exception:
            return locale.getdefaultlocale()[1]


def verify_and_correct_box_or_none(lp, box, data_format, img_w, img_h):
    error_msg = f'{lp} Illegal box [{", ".join([str(x) for x in box])}], img wxh: {img_w}, {img_h}'
    if len([x for x in box if x < 0]) > 0:
        logger.error(f'{error_msg}. Skip this box.')
        return None

    if data_format == TSV_FORMAT_LTWH_NORM:
        box[2] = int((box[0] + box[2]) * img_w)
        box[3] = int((box[1] + box[3]) * img_h)
        box[0] = int(box[0] * img_w)
        box[1] = int(box[1] * img_h)

    boundary_ratio_limit = 1.02
    if box[0] >= img_w or box[1] >= img_h or box[2] / img_w > boundary_ratio_limit \
            or box[3] / img_h > boundary_ratio_limit or box[0] >= box[2] or box[1] >= box[3]:
        logger.error(f'{error_msg}. Skip this box.')
        return None

    box[2] = min(box[2], img_w)
    box[3] = min(box[3], img_h)

    return box
