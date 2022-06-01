from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from PIL import Image
import csv
import pandas as pd
import sys
import os.path
import utils as uu


class eu_moths(Dataset):
    image_folder = '/images/'

    def __init__(self, root_dir, transform=None, target_transform=None, datasplit=None, label='biological'):
        # label:
        #   name: biological name
        #   engl: english translation
        #       only 198 classes - two with no translation
        #           + ecliptopera capitata (71 - training_set)
        #           + pungeleria capreolaria (168 - test_set)
        #   number: labelnumber

        self.root_dir = root_dir  # root for project
        self.path_engl = os.path.join(
            self.root_dir, "data-set-extensions/eu_moth/biological_to_english.txt")
        self.root = uu.get_root_eu_moths(self.root_dir)  # path to dataset
        self.preprocess = transform  # should be clip preprocess
        self.target_transform = target_transform
        self.loader = default_loader
        self.datasplit = datasplit
        self.label = label
        self.filepath = []
        self.labels = []
        self.label_names = []
        self.classes = []
        self.split = []
        self.label_engl = []  # all entries
        self.label_mapping = {}  # all classes
        self.class_names = {}
        self.classes_biological = []
        self.classes_english = []
        self.data = None

        self._load_data()
        self._fill_label_names()

        if self.label == 'english':
            self._fill_label_engl()

        self._load_dataframe()
        self._edit_classes()

        print(
            f"An example of the label looks like: {self.__getitem__(0)[1]}"
        )

    def _get_class_labels():
        pass

    def _load_data(self):
        with open(self.root + '/images.txt', 'r') as fd:
            reader = csv.reader(fd)
            for i in reader:
                ind, fp = str(i[0]).split(' ')
                self.filepath.append(fp)

        with open(self.root + '/labels.txt', 'r') as fd:
            reader = csv.reader(fd)
            for i in reader:
                self.labels.append(int(i[0]))

        with open(self.root + '/tr_ID.txt', 'r') as fd:
            reader = csv.reader(fd)
            for i, ids in enumerate(reader):
                self.split.append(int(ids[0]))

        with open(self.root + '/class_names.txt', 'r') as fd:
            reader = csv.reader(fd)
            for i, class_names in enumerate(reader):
                self.classes_biological += class_names
                self.class_names[i] = class_names[0]

        if self.label == 'english' and (self.path_engl is not None):
            with open(self.path_engl, 'r') as fd:
                reader = csv.reader(fd)
                for i in reader:
                    biolocical_label, english_label = i[0].split(';')
                    self.classes_english.append(english_label.lower())
                    self.label_mapping[biolocical_label] = english_label.lower(
                    )

            self.classes = self.classes_english
        else:
            self.classes = self.classes_biological

    def _fill_label_names(self):
        for item in self.labels:
            self.label_names.append(
                self._normalize_text(self.class_names[item]))

    def _fill_label_engl(self):
        for item in self.label_names:
            self.label_engl.append(self.label_mapping[item])

    def _edit_classes(self):
        for i in range(len(self.classes)):
            self.classes[i] = self._normalize_text(self.classes[i])

    def _normalize_text(self, text):
        return text.replace('_', ' ').replace(' - ', '/').lower()

    def _load_dataframe(self):

        if self.label == 'english':
            self.data = pd.DataFrame({
                "filepath": self.filepath,
                "label": self.labels,
                "split": self.split,
                "label_name": self.label_names,
                "label_engl": self.label_engl
            })

            for species in ['ecliptopera capitata', 'pungeleria capreolaria']:
                self.data.drop(
                    self.data[self.data['label_name'] == species].index, inplace=True)
        else:
            self.data = pd.DataFrame({
                "filepath": self.filepath,
                "label": self.labels,
                "split": self.split,
                "label_name": self.label_names
            })

        if self.datasplit == 0 or self.datasplit == 'test':
            self.data = self.data[self.data.split == 0]
        elif self.datasplit == 1 or self.datasplit == 'val':
            self.data = self.data[self.data.split == 1]
        elif self.datasplit == 2:
            self.data = self.data[self.data.split == 2]
        elif self.datasplit == 3:
            self.data = self.data[self.data.split == 3]
        elif self.datasplit == 'train':
            self.data = self.data[(self.data.split == 2)
                                  | (self.data.split == 3)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = self.root + self.image_folder + \
            self.data['filepath'].iloc[idx]

        img = self.loader(img_path)
        if self.preprocess is not None:
            img = self.preprocess(img)

        if self.target_transform is not None:
            label_name = self.data['label_name'].iloc[idx]
            label = self.target_transform(label_name)
        else:
            if self.label == 'biological':
                label = self.data['label_name'].iloc[idx]
            elif self.label == 'number':
                label = self.data['label'].iloc[idx]
            elif self.label == 'english':
                label = self.data['label_engl'].iloc[idx]
            else:
                print(
                    'Error: label-parameter: biological (biological), number or engl (english translation)')
        return img, label
