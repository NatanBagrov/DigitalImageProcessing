import logging
import imghdr
import os

from tqdm import tqdm
from PIL import Image


def remove_if_not_image(truth_file_path):
    if 'jpeg' != imghdr.what(truth_file_path):
        logger.error('%s is not image', truth_file_path)
        os.remove(truth_file_path)
    else:
        try:
            img = Image.open(truth_file_path)
            img.convert('RGB')
        except OSError:
            logger.exception('%s is problematic', truth_file_path)

            try:
                os.remove(truth_file_path)
            except FileNotFoundError:
                logger.exception('Race condition on %s', truth_file_path)


NUMBER_OF_PROCESSES = 16
LOG_FILE_PATH = 'data/log.log'
MAPPING_FILE_PATH = 'data/tmp/fall11_urls.txt'
DATA_ROOT = 'data'

logger = logging.getLogger('')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)
logger.addHandler(stream_handler)
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    truth_image_root = os.path.join(DATA_ROOT, 'ImageNet')

    for set_name in ('test', 'train'):
        truth_set_image_root = os.path.join(truth_image_root, set_name)

        assert os.path.isdir(truth_set_image_root)
        os.makedirs(truth_set_image_root, exist_ok=True)

        for class_name in tqdm(os.listdir(truth_set_image_root)):
            truth_class_root = os.path.join(truth_set_image_root, class_name)

            if (class_name.startswith('n') or class_name == 'test') and os.path.isdir(truth_class_root):
                assert os.path.isdir(truth_class_root)
                os.makedirs(truth_class_root, exist_ok=True)

                for truth_file_name in os.listdir(truth_class_root):
                    truth_file_path = os.path.join(truth_class_root, truth_file_name)
                    remove_if_not_image(truth_file_path)

