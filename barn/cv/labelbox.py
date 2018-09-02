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

    @staticmethod
    def create_mask_from_label(label_name, json_record, img_size=(120, 160)):
        label_data = json_record['Label'][label_name]
        mask = Image.fromarray(np.zeros(img_size), mode='L')

        for geometry in label_data:
            poly = create_polygon_tuple(geometry['geometry'])
            ImageDraw.Draw(mask).polygon(poly, fill=255)
        return mask