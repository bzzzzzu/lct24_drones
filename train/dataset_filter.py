import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

# Объекты больше чем 10% кадра как правило или уже упали на объект или какие-то вообще левые картинки;
# Возможно будет немного хуже по метрикам, но должно более точно соответствовать задаче;
# Код использовался как часть чистки датасета

labels_size = [0] * 100

for f in os.listdir('datasets/drones/labels'):
    with open(f'datasets/drones/labels/{f}', 'r') as ftext:
        for line in ftext.readlines():
            l = str.strip(line)
            max_size = max(float(str.split(l, ' ')[3]), float(str.split(l, ' ')[4]))
            if max_size < 0.10:
                # JPEG, jpg, png
                if os.path.exists(f'datasets/drones/images/{str.split(f, ".txt")[0]}.JPEG'):
                    shutil.copy(f'datasets/drones/images/{str.split(f, ".txt")[0]}.JPEG', f'datasets/drones_clean/images/{str.split(f, ".txt")[0]}.JPEG')
                if os.path.exists(f'datasets/drones/images/{str.split(f, ".txt")[0]}.jpg'):
                    shutil.copy(f'datasets/drones/images/{str.split(f, ".txt")[0]}.jpg', f'datasets/drones_clean/images/{str.split(f, ".txt")[0]}.jpg')
                if os.path.exists(f'datasets/drones/images/{str.split(f, ".txt")[0]}.png'):
                    shutil.copy(f'datasets/drones/images/{str.split(f, ".txt")[0]}.png', f'datasets/drones_clean/images/{str.split(f, ".txt")[0]}.png')
            labels_size[int(max_size*100-0.000001)] = labels_size[int(max_size*100-0.000001)] + 1

plt.plot(np.arange(len(labels_size)), labels_size)
plt.ylim([0, np.max(labels_size) + 100])
plt.xlim([0, len(labels_size)])
plt.grid()
plt.show()