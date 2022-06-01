import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from PIL import Image

# from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class Cub2011(Dataset):
    base_folder = "CUB_200_2011/images"

    class_path = "/CUB_200_2011/classes.txt"
    attributes_path = "/attributes.txt"
    parts_path = "/CUB_200_2011/parts/parts.txt"

    url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        root,
        train=True,
        transform_image=None,
        transform_label=None,

        loader=default_loader,
        download=False,
        label_mapping=True,
        label="class",
        preprocess=None
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform_image
        self.target_transform = transform_label
        self.loader = default_loader

        self.train = train
        self.preprocess = preprocess  # Clip preprocess from Clip.load
        self.label = label
        self.label_mapping = label_mapping

        (
            self.class_map,
            self.attribute_map,
            self.part_map,
            self.certain_map,
        ) = self._label_maps()

        self.classes = self._classes(self.class_map)

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        if self.loader is None and (transform_image is not None or preprocess is not None):
            print("error: you can't set loader=None and preprocess or transform the image - loader will be set default")
            self.loader = default_loader

        print(
            f"You choose '{self.label}' label. An example looks like: {self.__getitem__(0)[1]}"
        )

    def _get_sample(self, idx):
        """
        Form der Labels, die zurückgegeben werden

        Varianten: 
        class: Klassenlabel
        att: Liste aller Attribute, die auf dem Bild zu sehen sind
        parts: Liste aller Parts, die auf dem Bild zu sehen sind
        class + att: Liste mit Klassenlabel und att
        class + parts: Liste mit Klassenlabel und parts
        """

        sample = self.data.iloc[idx]

        if self.label == "att":
            sample_label = sample.attribute_ids

            if self.label_mapping is True:
                sample_label = [self.attribute_map[x] for x in sample_label]

        elif self.label == "parts":
            sample_label = sample.part_ids

            if self.label_mapping is True:
                sample_label = [self.part_map[x] for x in sample_label]

        elif self.label == "class + parts":
            sample_label = [sample.img_class] + sample.part_ids

            if self.label_mapping is True:
                sample_label = [self.class_map[sample_label[0]]] + [
                    self.part_map[x] for x in sample_label[1:]
                ]

        elif self.label == "class + att":
            sample_label = [sample.img_class] + sample.attribute_ids

            if self.label_mapping is True:
                sample_label = [self.class_map[sample_label[0]]] + [
                    self.attribute_map[x] for x in sample_label[1:]
                ]

        else:  # class as default
            sample_label = sample.img_class

            if self.label_mapping is True:
                sample_label = self.class_map[sample_label]

        return {"label": sample_label, "filepath": sample.filepath}

    def _load_data(self):
        images = self._load_dataframe("images.txt", ["img_id", "filepath"])
        image_class_labels = self._load_dataframe(
            "image_class_labels.txt", ["img_id", "img_class"]
        )
        image_attribute_labels = self._load_dataframe(
            "attributes/image_attribute_labels.txt",
            ["img_id", "attribute_id", "is_present", "certainly_id", "time"],
        )
        image_attribute_labels = image_attribute_labels[
            (image_attribute_labels.is_present != 0)
        ]
        image_attribute_labels = (
            image_attribute_labels.groupby("img_id")["attribute_id"]
            .apply(list)
            .reset_index(name="attribute_ids")
        )
        image_part_labels = self._load_dataframe(
            "parts/part_locs.txt", ["img_id", "part_id", "x", "y", "visible"]
        )
        image_part_labels = image_part_labels[(image_part_labels.visible != 0)]
        image_part_labels = (
            image_part_labels.groupby("img_id")["part_id"]
            .apply(list)
            .reset_index(name="part_ids")
        )

        train_test_split = self._load_dataframe(
            "train_test_split.txt", ["img_id", "is_training_img"]
        )

        data = images.merge(image_class_labels, on="img_id")
        data = data.merge(image_attribute_labels, on="img_id")
        data = data.merge(image_part_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        if self.train is not None:
            if self.train == True:
                self.data = self.data[self.data.is_training_img == 1]
            elif self.train == False:
                self.data = self.data[self.data.is_training_img == 0]

    def _load_dataframe(self, filepath, list_col_names):
        df = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", filepath),
            sep=" ",
            names=list_col_names,
        )
        return df

    def _check_integrity(self):
        try:
            self._load_data()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        # doesnt work
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

        # alternative approach does not work either
        # download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self._get_sample(idx)
        path = os.path.join(self.root, self.base_folder, sample["filepath"])
        label = sample["label"]

        if self.loader is not None:
            img = self.loader(path)
        else:
            img = path

        if self.transform is not None:
            img = self.transform(img)

        if self.preprocess is not None:
            # .unsqueeze(0)  # preprocess from clip.load
            img = self.preprocess(img)

        if self.target_transform is not None:
            if self.label_mapping == False:
                print(
                    "Warnung: Label Mapping False aber Label Transformation gewünscht. Überprüfe ob Label sinnvoll."
                )
            label = self.target_transform(label)

        return img, label

    def _label_maps(self):
        class_names = pd.read_csv(
            self.root + self.class_path, sep=".", header=None, names=["number", "name"]
        )

        attribute_names = pd.read_csv(
            self.root + self.attributes_path, sep=" ", header=None, names=["attr"]
        )
        attribute_names = pd.DataFrame(
            attribute_names["attr"]
            .str.split("::", 1, expand=True)
            .rename(columns={0: "attr", 1: "value"})
        )

        self._clean_text(class_names, "name")
        self._clean_text(attribute_names, "attr")
        self._clean_text(attribute_names, "value")

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

        certain_map = {1: "not visible", 2: "guessing",
                       3: "probably", 4: "definitely"}

        for i, name in enumerate(class_names["name"]):
            d = {i + 1: name}
            class_map.update(d)

        for i, val in enumerate(zip(attribute_names["attr"], attribute_names["value"])):
            attr, value = val
            d = {i + 1: [attr, value]}
            attribute_map.update(d)

        return class_map, attribute_map, part_map, certain_map

    def _clean_text(self, df, col):
        df[col] = (
            df[col]
            .str.replace(r"_", " ")
            .replace(r"\(.*?\)", " ", regex=True)
            .str.strip()
        )
        return df

    def _classes(self, class_map):
        return list(class_map.values())
