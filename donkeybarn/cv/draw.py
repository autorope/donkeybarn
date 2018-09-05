import numpy as np
from PIL import Image, ImageDraw


def line_from_coef(img, coef, color=(255,0,0)):
    img_height = img.shape[0]

    # create function from coefficients and calculate the
    # x value of the line for each y pixel
    p = np.poly1d(coef)
    y = np.arange(0, img_height)
    x = (p(np.arange(0, img_height))).astype(np.int)
    crds = np.stack([x, y], axis=1)

    # convert the coordinates into a form accepted by PIL
    crds_list = [(i[0], i[1]) for i in crds]

    # draw the line on the image
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    draw.line(crds_list, fill=color, width=3)
    return np.array(im)


