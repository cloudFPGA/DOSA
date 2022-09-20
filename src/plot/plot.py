import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_result(losses, accuracies, export_path=None):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(losses, color='tab:blue')
    axs[1].plot(accuracies, color='tab:red')
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.show()

    if export_path is not None:
        fig.savefig(export_path)
