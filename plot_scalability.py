import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)

cores = np.array([1, 2, 4, 8, 16])
times = np.array([2936,  1341,  734,  443, 289])

fig = plt.figure()
ax = plt.axes()

ax.plot(cores, cores, 'k--')
ax.plot(cores, np.divide(times[0],times), 'r-d')
ax.set_xlim((1,16))
ax.set_ylim((1,16))
ax.set_xlabel('nodes')
ax.set_ylabel('speedup')
ax.set_aspect('equal', 'box')
plt.show()
