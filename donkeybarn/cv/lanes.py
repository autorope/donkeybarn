
import numpy as np


def find_line_coef_from_mask(mask, window_height=10):
    img_height = mask.shape[0]
    lane_cords = []
    y_window_top = 0

    while y_window_top < img_height:
        y_window_bottom = int(y_window_top)
        y_window_top = min(img_height, y_window_top + window_height)
        histogram = np.sum(mask[img_height - y_window_top:img_height - y_window_bottom, :], axis=0)
        if sum(histogram) > 0:
            y = int(y_window_top - window_height / 2)
            x = np.argmax(histogram)
            lane_cords.append((x, img_height - y))
        else:
            next

    lane_cord_arr = np.array(lane_cords)
    line_coef = np.polyfit(lane_cord_arr[:, 1], lane_cord_arr[:, 0], deg=2)

    return line_coef