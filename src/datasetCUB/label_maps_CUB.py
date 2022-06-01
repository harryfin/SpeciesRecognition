import os.path
import pandas as pd


def label_maps(root):
    class_path = "/CUB_200_2011/classes.txt"
    attributes_path = "/attributes.txt"
    parts_path = "/CUB_200_2011/parts/parts.txt"

    if not _check_files([root + class_path, root + attributes_path, root + parts_path]):
        return 0, 0, 0, 0

    class_names = pd.read_csv(
        root + class_path, sep=".", header=None, names=["number", "name"]
    )
    attribute_names = pd.read_csv(
        root + attributes_path, sep=" ", header=None, names=["attr"]
    )
    attribute_names = pd.DataFrame(
        attribute_names["attr"]
        .str.split("::", 1, expand=True)
        .rename(columns={0: "attr", 1: "value"})
    )
    attribute_names["value"] = (
        attribute_names["value"]
        .str.replace(r"_", " ")
        .replace(r"\(.*?\)", " ", regex=True)
        .str.strip()
    )

    class_map = {}
    attribute_map = {}
    part_map = {
        1: "back",
        2: "beak",
        3: "belly",
        4: "breast",
        5: "crown",
        6: "forehead",
        7: "left eye",
        8: "left leg",
        9: "left wing",
        10: "nape",
        11: "right eye",
        12: "right leg",
        13: "right wing",
        14: "tail",
        15: "throat",
    }

    certain_map = {1: "not visible", 2: "guessing", 3: "probably", 4: "definitely"}

    for i, name in enumerate(class_names["name"]):
        d = {i + 1: name}
        class_map.update(d)

    for i, val in enumerate(zip(attribute_names["attr"], attribute_names["value"])):
        attr, value = val
        d = {i + 1: [attr, value]}
        attribute_map.update(d)

    return class_map, attribute_map, part_map, certain_map


def _check_file(path):
    if os.path.isfile(path):
        return True
    else:
        print("Error: {} not valid".format(path))
        return False


def _check_files(list_of_paths):
    # list_of_paht [class_path, attributes_path, parts_path]
    for path in list_of_paths:
        try:
            os.stat(path)
        except OSError:
            print("Error: {} not valid".format(path))
            return False
        except:
            print("Error")
            return False
        else:
            print("path ok")  # code for try
            return True


def map_label(dict, index):
    return dict[index]
