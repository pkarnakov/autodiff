#!/usr/bin/env python3

from odil import plotutil
import matplotlib.pyplot as plt
plotutil.set_extlist(['svg'])


def save_arrow(path, lx=None, ly=None, node=None, xlabel=None, ylabel=None):
    figsize = (lx, 0.06 if ly is None else ly)
    fig, ax = plt.subplots(figsize=figsize)
    hw = 0.08
    hwh = hw * 0.5
    if node is None:
        node = bool(lx and ly)

    ax.set_xlim(-hwh if node or ly else 0, lx or hwh)
    ax.set_ylim(-hwh if node or lx else 0, ly or hwh)
    ax.set_aspect('equal')
    ax.set_axis_off()

    def arrow(x, y, dx, dy):
        ax.arrow(x,
                 y,
                 dx,
                 dy,
                 width=0.015,
                 head_width=hw,
                 overhang=0.2,
                 lw=0,
                 facecolor='k',
                 edgecolor='none',
                 length_includes_head=True)

    if lx:
        arrow(0, 0, lx, 0)
    if ly:
        arrow(0, 0, 0, ly)
    if node:
        ax.scatter(0, 0, edgecolor='none', facecolor='k', s=10)
    if xlabel:
        ax.text(lx * 0.6, hw * 0.8, xlabel, fontsize=14)
    if ylabel:
        ax.text(hw * 0.7, ly * 0.6, ylabel, fontsize=14)

    plotutil.savefig(fig,
                     path,
                     bbox_inches='tight',
                     transparent=True,
                     pad_inches=0.01)


if __name__ == "__main__":
    save_arrow('axes_xt', 0.4, 0.4, xlabel='x', ylabel='t')
