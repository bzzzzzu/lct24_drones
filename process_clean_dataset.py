import os
import shutil

list_of_files = os.listdir('datasets/drones_clean/images/')

with open('clean_list.txt', 'w', encoding='utf-8') as clf:
    for l in list_of_files:
        clf.writelines(f'{l}\n')

for f in list_of_files:
    label_name = f"{'.'.join(str.split(f, '.')[:-1])}.txt"
    shutil.copy(f'datasets/drones/labels/{label_name}', f'datasets/drones_clean/labels/{label_name}')