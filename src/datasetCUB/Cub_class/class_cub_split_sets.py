import sys
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from datasetCUB.Cub_class.class_cub import Cub2011


class class_cub_split_sets(Dataset):
    root = "~/Datasets/CUB2011"
    splits = [
        "allclasses",
        "testclasses",
        "trainclasses1",
        "trainclasses2",
        "trainclasses3",
        "trainvalclasses",
        "valclasses1",
        "valclasses2",
        "valclasses3",
    ]

    def __init__(
        self, split_class, preprocess, cub=None, label_mapping=False
    ):

        self.split_class = split_class
        self._check_split_class()

        self.preprocess = preprocess
        self.loader = default_loader

        self.label_mapping = label_mapping

        if cub is not None:
            self.cub_dataset = cub
        else:
            self.cub_dataset = self._get_cub_data()
        self.all_cub_classeses = self.cub_dataset.classes
        self.classes = None

        self.split_data = self._load_dataset_with_split()
        self.filepath = self.split_data["filepath"].tolist()
        # self.filepath = self.filepath.tolist()
        self.img_class = self.split_data["img_class"].tolist()
        # self.img_class = self.img_class.tolist()
        self.is_training = self.split_data["is_training_img"].tolist()

    def _check_split_class(self):
        if self.split_class not in self.splits:
            print("false split choosen: " + self.split_class)
            sys.exit()

    def _get_cub_data(self):
        return Cub2011(root=self.root, train=None)

    def _load_dataset_with_split(self):
        self.classes = self._load_splits()
        df_set = self.cub_dataset.data.query("img_class == @self.classes")

        # get class names (List starts at Zero - so minus 1)
        for i in range(len(self.classes)):
            self.classes[i] = self.all_cub_classeses[self.classes[i] - 1]
        return df_set

    def _load_splits(self):
        # Proposed Split Version 2.0 from:
        # https://www.mpi-inf.mpg.de/de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly

        root = "~/Coding/SpeciesRecognition/data-set-extensions/Zero-Shot-Split-Sets/xlsa17/data/CUB/"

        choice = [
            "allclasses",
            "testclasses",
            "trainclasses1",
            "trainclasses2",
            "trainclasses3",
            "trainvalclasses",
            "valclasses1",
            "valclasses2",
            "valclasses3",
        ]

        if self.split_class in choice:

            with open(root + self.split_class + ".txt") as f:
                lines = f.readlines()

            for ind, line in enumerate(lines):
                lines[ind] = line.replace("\n", "")
                lines[ind], _ = line.split(".")
                lines[ind] = int(lines[ind])

            return lines
        else:
            print("No correct split_class - choose from", choice)
            return None

    def __len__(self):
        return len(self.img_class)

    def __getitem__(self, idx):

        img_path = self.filepath[idx]
        root = "~/Datasets/CUB2011/CUB_200_2011/images/"
        img = root + img_path
        img = self.preprocess(self.loader(img))

        label = self.img_class[idx]
        label = label - 1  # mapping list is starting with 0 and labels with 1

        if self.label_mapping:
            label = self.all_cub_classeses[label]

        return img, label


# Load Dataloader
# my_dataset = class_cub_split_sets('test_classes', preprocess #from CLIP#)
# dataloader = DataLoader(my_dataset, batch_size = BATCH_SIZE)
