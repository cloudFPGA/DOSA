#  /*******************************************************************************
#   * Copyright 2022 -- 2024 IBM Corporation
#   *
#   * Licensed under the Apache License, Version 2.0 (the "License");
#   * you may not use this file except in compliance with the License.
#   * You may obtain a copy of the License at
#   *
#   *     http://www.apache.org/licenses/LICENSE-2.0
#   *
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS,
#   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   * See the License for the specific language governing permissions and
#   * limitations under the License.
#  *******************************************************************************/
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from matplotlib import colors


def plot_training_evolution(losses, accuracies, export_path=None):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(losses, color='tab:blue')
    axs[1].plot(accuracies, color='tab:red')
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.show()

    if export_path is not None:
        fig.savefig(export_path)


def plot_values_distribution(values, export_path=None, num_bins=20, show_symmetric=True):
    fig, axs = plt.subplots()
    N, bins, patches = axs.hist(values, bins=num_bins)

    # colors
    fracs = N / N.max()
    norm = colors.Normalize(0, fracs.max())
    for frac, patch in zip(fracs, patches):
        color = plt.cm.viridis(norm(frac))
        patch.set_facecolor(color)

    if show_symmetric:
        max_abs = abs(values).max()
        axs.set_xlim([-max_abs, max_abs])
    fig.show()

    if export_path is not None:
        fig.savefig(export_path)
