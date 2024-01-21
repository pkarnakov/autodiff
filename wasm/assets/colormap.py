#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

cmap = mpl.colormaps['PuOr_r']

fig, ax = plt.subplots()
values = np.linspace(0, 1, 101)
colors = (np.array([cmap(x)[:3] for x in values]) * 255).astype(np.uint8)
ax.set_axis_off()
ax.imshow(colors[None, :, :], aspect=10)
fig.savefig("colormap.png", bbox_inches='tight', pad_inches=0)

with open('colormap.h', 'w') as f:
    f.write("const uint8_t kColorsRGB[][3] = {\n")
    for c in colors:
        f.write("    {{{}, {}, {}}},\n".format(*c))
    f.write("};\n")

