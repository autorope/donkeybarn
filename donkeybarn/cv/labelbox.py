import json
from PIL import Image, ImageDraw
import numpy as np


def create_polygon_tuple(pts_list, img_height):
    pt_array = []
    for pt in pts_list:
        pt_array.append((pt['x'], img_height - pt['y']))
    return pt_array


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