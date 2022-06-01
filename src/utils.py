import numpy as np
import os.path


def unpack(list_in):
    list_out = []
    for i in list_out:
        list_out.append(i[0])
    return list_out


class config_structure:
    # change dict with entries to class with attributes
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_dataset_path(root, dataset_path):
    path = os.path.join(root, dataset_path)

    try:
        f = open(path, "r")
        path = f.read()
    except OSError:
        print('cannot open', path)

    return path


def get_root_CUB(root):
    return get_dataset_path(root, 'root_directories/root_CUB.txt')


def get_root_eu_moths(root):
    return get_dataset_path(root, 'root_directories/root_eu-moths.txt')


def get_root_mmc(root):
    return get_dataset_path(root, 'root_directories/root_mmc.txt')


def write_acc_file(acc, filename):
    # acc - list of results
    std = str(np.std(acc))
    mean = str(np.mean(acc))
    file = filename + ".txt"

    with open(file, "w") as f:
        f.write("acc,")
        for i in acc:
            text = str(i) + ","
            f.write(text)
        f.write("\n")
        text = "mean," + mean + ",\n"
        f.write(text)
        text = "std," + std + ",\n"
        f.write(text)


def write_acc_file(acc, filename):
    # acc - list of results
    std = str(np.std(acc))
    mean = str(np.mean(acc))
    file = filename + ".txt"

    with open(file, "w") as f:
        f.write("acc,")
        for i in acc:
            text = str(i) + ","
            f.write(text)
        f.write("\n")
        text = "mean," + mean + ",\n"
        f.write(text)
        text = "std," + std + ",\n"
        f.write(text)


def get_root(poject_folder="SpeciesRecognition"):
    # is searching the root path of the project folder
    path = os.path.dirname(os.path.abspath(__file__))
    while True:
        path_parent, folder = os.path.split(path)
        if folder != poject_folder and path_parent != '/':
            path = path_parent
        elif path_parent == '/':
            break
        else:
            break
    return(path)
