from matplotlib import pyplot as plt
import numpy as np

def imgs(imgs_dict, cmap=None):
    """
    Convenience function to show many images in a single plot.
    """

    plot_count = len(imgs_dict)

    fig, ax = plt.subplots(1, plot_count, figsize=(15, 6), squeeze=False)

    axoff = np.vectorize(lambda ax: ax.axis('off'))
    axoff(ax)

    for i, data in enumerate(imgs_dict):
        ax[0, i].imshow(data['array'], cmap=cmap)
        ax[0, i].set_title(data.get('name', ' '))

    return fig