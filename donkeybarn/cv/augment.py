import random
import cv2
import numpy as np
from PIL import Image

class BaseProcessor:

    def __init__(self):
        # put your initialization logic here
        pass

    def run_img(self, img):
        arr = np.array(img)
        new_arr = self.run_arr(arr)
        return Image.fromarray(new_arr)

    def run_arr(self, arr):
        # override this method with your transformation method.
        return arr

    def run(self, input_list):
        xp = self.run_arr(*input_list)
        return [xp]


class RandomRectangles(BaseProcessor):
    def __init__(self, max_rectangles=2):
        self.run_arr = self.add_rectangles

    @staticmethod
    def gen_random_rectangle_coords(top, bottom, left, right, min_width, max_width):
        width = int(random.randint(min_width, max_width) / 2)
        height = random.randint(30, 50)
        x_center = random.randint(left, right)
        y_center = random.randint(top, bottom)
        tl = (max(x_center - width, 0), y_center + height)
        br = (x_center + width, max(y_center - height, 0))
        return tl, br

    def add_rectangle(self, arr, probability=.2,
                      top=10, bottom=30, left=10, right=150,
                      min_width=10, max_width=30):
        tl, br = self.gen_random_rectangle_coords(top, bottom, left, right, min_width, max_width)
        color = tuple(random.choice(range(0, 200)) for _ in range(3))
        #print('arr.shape: {}, tl: {}, br: {}, color:{}'.format(arr.shape, tl, br, color))
        arr = cv2.rectangle(arr.copy(), tl, br, color, -1)
        return arr

    def add_rectangles(self, arr, n=2):
        for _ in range(n):
            arr = self.add_rectangle(arr, probability=.2)
        return arr


class RandomBrightness(BaseProcessor):
    def __init__(self, probability=.4):
        self.run_arr = self.random_brightness

    def random_brightness(self, arr, probability=.4):
        random_bright = np.random.uniform(.1, 1) + .5
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)  # convert it to hsv
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random_bright, 0, 255)
        arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return arr


class RandomBlur(BaseProcessor):

    def __init__(self, probability=.4):
        self.run_arr = self.random_blur

    def random_blur(self, arr, probability=.4, min_kernal_size=2, max_kernal_size=3):
        kernal_size = random.randint(min_kernal_size, max_kernal_size)
        kernel = np.ones((kernal_size, kernal_size), np.float32) / (kernal_size ** 2)
        arr = cv2.filter2D(arr, -1, kernel)
        return arr


class RandomFlip(BaseProcessor):
    def __init__(self, func_on_flip=lambda x: x * -1):
        self.func_on_flip = func_on_flip

    def run_arr(self, arr):
        x = np.fliplr(arr)
        return x.copy()

    def run(self, inputs):
        x = inputs[0]
        y = inputs[1]
        flip = True
        x = self.run_arr(x)
        y = self.func_on_flip(y)
        return [x, y]


def augment_images(arr):
    arr = add_rectangles(arr, n=4)
    arr = random_blur(arr, probability=.2, min_kernal_size=2, max_kernal_size=4)
    arr = random_brightness(arr)
    return arr
