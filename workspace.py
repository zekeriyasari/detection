import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 21, 5)
minor_ticks = np.arange(0, 21, 0.25)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

# and a corresponding grid

ax.grid(which='both')

# or if you want differnet settings for the grids:
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

plt.show()
