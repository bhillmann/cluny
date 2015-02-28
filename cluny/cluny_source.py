import os
import numpy as np

def generate_source():
    file_paths = []
    file_names = []
    data_path = os.path.abspath(os.path.join(os.curdir, '..', 'data'))
    for filename in os.listdir(data_path):
        if not 'dat' in filename:
            continue
        file_paths.append(os.path.join(data_path, filename))
        file_names.append(filename)
    for i, file_path in enumerate(file_paths):
        with open(file_path) as f:
            mat = np.loadtxt(f)
            print(file_names[i])
            yield mat
