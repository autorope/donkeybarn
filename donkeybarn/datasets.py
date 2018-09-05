import os
from os.path import dirname, abspath
import glob

from donkeybarn import fileio


class BaseDataset:

    url = None
    file_format = ".tar.gz"

    @classmethod
    def load(cls, data_dir=None):

        if data_dir is None:
            data_dir = os.path.expanduser('~/donkey_data')

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        filename = cls.__name__ + cls.file_format

        filepath = os.path.join(data_dir, filename)
        fileio.download_file(cls.url, filepath)
        extracted_folder = fileio.extract_file(filepath, data_dir)

        obj = cls(extracted_folder)
        return obj

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.img_paths = glob.glob(os.path.join(base_dir, '*.jpg'))
        self.img_paths = sorted(self.img_paths, key=lambda x: int(os.path.split(x)[-1].split('_')[0]))




class Donkey2CalibrationImages(BaseDataset):

    url = "https://drive.google.com/uc?export=download&id=1yk758anknZqAwPBcrWa4vGZ_3Xgh1gmU"
    file_format = ".tar.gz"
    checkerboard_size = (7, 9)


class AmericanSteelLabeled(BaseDataset):
    url = 'https://drive.google.com/uc?export=download&id=1GKkB_xMgOoUPf0J3OGzj6wtke1eqPU0Q'
    format = ".tar.gz"


if __name__ == "__main__":
    obj = Donkey2CalibrationImages.load()
    print('test')

    print(obj.img_paths)
