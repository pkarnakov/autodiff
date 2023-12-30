#!/usr/bin/env python3

import numpy as np
import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(path, u, vmin=None, vmax=None):
    vmin = u.min() if vmin is None else vmin
    vmax = u.max() if vmax is None else vmax
    nx, ny = u.shape
    fig = plt.figure(frameon=False)
    fig.set_size_inches(nx, ny)
    extent = [0, 1, 0, 1]
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    fig.add_axes(ax)
    ax.imshow(u.T, interpolation='nearest', cmap='Oranges', vmin=u.min(),
              vmax=u.max(), extent=extent, origin='lower', aspect='equal')
    fig.savefig(path, pad_inches=0, bbox_inches='tight', dpi=1)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Text file with 2D array")
    parser.add_argument('output', help="Output image")
    parser.add_argument('--vmin', type=float, default=0, help="Minimum value")
    parser.add_argument('--vmax', type=float, default=1, help="Maximum value")
    args = parser.parse_args()
    u = np.loadtxt(args.input)
    plot(args.output, u)
