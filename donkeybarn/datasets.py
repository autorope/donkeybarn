import os
from os.path import dirname, abspath
import glob

from donkeybarn import fileio
import json
from PIL import Image, ImageDraw
import numpy as np




#TODO: Fatten the labels datastructure in the dataset so that they are accessable from
# dataset.label[0]
# dataset.img[0]
# dataset[0, ['img']]
# merged = dataset1 + dataset2

class BaseDataset:
    base_dir = None
    img_paths = []
    labels = None


class LoadedDataset(BaseDataset):

    url = None
    file_format = ".tar.gz"

    @classmethod
    def load(cls, data_dir=None):

        if data_dir is None:
            data_dir = cls.get_data_dir()

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        filename = cls.__name__ + cls.file_format

        filepath = os.path.join(data_dir, filename)
        fileio.download_file(cls.url, filepath)
        extracted_folder = fileio.extract_file(filepath, data_dir)

        obj = cls(extracted_folder)
        return obj

    @classmethod
    def get_data_dir(cls, ):
        from donkeybarn import BARN_DATA_DIR
        return BARN_DATA_DIR

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.img_paths = glob.glob(os.path.join(base_dir, '*.jpg'))
        self.img_paths = sorted(self.img_paths, key=lambda x: int(os.path.split(x)[-1].split('_')[0]))

        try:
            labels_path = os.path.join(base_dir, 'labels.json')
            self.labels = LabelBoxData(labels_path)
            self.labels.gen_external_key_index()

        except FileNotFoundError as e:
            print('could not filed labels.json in {}, not loading labels.'.format(base_dir))




class LabelBoxData:

    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        sorted(self.data, key=lambda x: int(x["External ID"].split('_')[0]))

    def gen_external_key_index(self):
        self.key_index = {}
        for i, rec in enumerate(self.data):
            self.key_index[rec['External ID']] = i


    def get_mask_from_key(self, key, label_name):
        ix = self.key_index[key]
        rec = self.data[ix]
        mask = self.create_mask_from_label(label_name, rec)

        return mask


    @staticmethod
    def create_mask_from_label(label_name, rec, img_size=(120, 160)):
        label_data = rec['Label'][label_name]
        mask = Image.fromarray(np.zeros(img_size), mode='L')

        for geometry in label_data:
            poly = create_polygon_tuple(geometry['geometry'], img_size[0])
            ImageDraw.Draw(mask).polygon(poly, fill=255)
        return mask


def create_polygon_tuple(pts_list, img_height):
    pt_array = []
    for pt in pts_list:
        pt_array.append((pt['x'], img_height - pt['y']))
    return pt_array




class Donkey2CalibrationImages(LoadedDataset):

    url = "https://drive.google.com/uc?export=download&id=1yk758anknZqAwPBcrWa4vGZ_3Xgh1gmU"
    file_format = ".tar.gz"
    checkerboard_size = (7, 9)


class AmericanSteelLabeled(LoadedDataset):
    url = 'https://drive.google.com/uc?export=download&id=1GKkB_xMgOoUPf0J3OGzj6wtke1eqPU0Q'
    format = ".tar.gz"


class DriveaiLabeled(LoadedDataset):
    url = 'https://drive.google.com/uc?export=download&id=10R8VOHyzd0QD0zNLzLel5Mg6EWFOKSMX'
    format = ".tar.gz"


class MakerFaireLabeled(LoadedDataset):
    url = 'https://drive.google.com/uc?export=download&id=1ohTZYbuQwxLb63uZTajlDNn8cJmoG_az'
    format = ".tar.gz"


class AWSTrack(LoadedDataset):
    url = 'https://drive.google.com/uc?export=download&id=1h1zu6VN_txhyb86hHx3K6COXs_UqMIRL'
    format = ".tar.gz"

if __name__ == "__main__":
    obj = Donkey2CalibrationImages.load()
    print('test')

    print(obj.img_paths)