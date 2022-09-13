import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_loss(loss, export_path=None):
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(loss)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('loss')
    fig.show()

    if export_path is not None:
        fig.savefig(export_path)
