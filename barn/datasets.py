import os
from os.path import dirname, abspath
import glob

class DonkeyCalibration:

    @classmethod
    def load(cls):

        barn_path = dirname(dirname(abspath(__file__)))
        data_path = os.path.join(barn_path, 'data', 'camera_calibration')
        img_list = glob.glob(os.path.join(data_path, '*.jpg'))

        obj = DonkeyCalibration(img_list)
        return obj

    def __init__(self, img_paths):
        self.img_paths = img_paths



if __name__ == "__main__":
    obj = DonkeyCalibration.load()

    print('test')

    print(obj.img_paths)
