import os
import matplotlib.pyplot as plt
import numpy as np

labels_size = [0] * 100

for f in os.listdir('labels'):
    with open(f'labels/{f}', 'r') as ftext:
        for line in ftext.readlines():
            l = str.strip(line)
            max_size = max(float(str.split(l, ' ')[3]), float(str.split(l, ' ')[4]))
            #print(max_size*100)
            labels_size[int(max_size*100-0.000001)] = labels_size[int(max_size*100-0.000001)] + 1

plt.plot(np.arange(len(labels_size)), labels_size)
plt.ylim([0, np.max(labels_size) + 100])
plt.xlim([0, len(labels_size)])
plt.grid()
plt.show()