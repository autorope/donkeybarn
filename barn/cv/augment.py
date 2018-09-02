import random
import cv2
import numpy as np


def gen_random_rectangle_coords(top, bottom, left, right, min_width, max_width):
    width = int(random.randint(min_width, max_width) / 2)
    height = random.randint(30, 50)
    x_center = random.randint(left, right)
    y_center = random.randint(top, bottom)
    tl = (x_center - width, y_center + height)
    br = (x_center + width, y_center - height)
    return tl, br


def add_rectangle(arr, probability=.2,
                  top=10, bottom=30, left=10, right=150,
                  min_width=10, max_width=30):
    if probability > random.random():
        tl, br = gen_random_rectangle_coords(top, bottom, left, right, min_width, max_width)
        color = tuple(random.choice(range(0, 200)) for _ in range(3))
        arr = cv2.rectangle(arr, tl, br, color, -1)
    return arr


def add_rectangles(arr, n=2):
    for _ in range(n):
        arr = add_rectangle(arr, probability=.2)
    return arr


def random_blur(arr, probability=.2, min_kernal_size=2, max_kernal_size=3):
    if probability > random.random():
        kernal_size = random.randint(min_kernal_size, max_kernal_size)
        kernel = np.ones((kernal_size, kernal_size), np.float32) / (kernal_size ** 2)
        arr = cv2.filter2D(arr, -1, kernel)
    return arr


def random_brightness(arr, probability=.1):
    if probability > random.random():
        random_bright = np.random.uniform(.1, 1) + .5
        hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)  # convert it to hsv
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random_bright, 0, 255)
        arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return arr


def augment_images(arr):
    arr = add_rectangles(arr, n=5)
    arr = random_blur(arr, probability=.2, min_kernal_size=2, max_kernal_size=4)
    arr = random_brightness(arr)
    return arr
