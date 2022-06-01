from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from PIL import Image
import csv
import pandas as pd


class mmc_moths(Dataset):
    image_folder = '/images/'
    def __init__(self, root, transform=None, target_transform=None, datasplit=None, label='name'):


        self.root = root
        self.preprocess = transform # should be clip preprocess
        self.target_transform = target_transform
        self.loader = default_loader
        self.datasplit = datasplit
        self.label = label
        self.filepath = []
        self.labels = []
        self.label_names = []   
        self.classes = []     
        self.split = []
        self.class_names = {}
        
        self._load_data()
        self._fill_label_names()
        
        self.data = None
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
 
        with open(self.root + '/class_names.txt', 'r') as fd:
            reader = csv.reader(fd)
            for i, class_names in enumerate(reader):
                self.classes += class_names
                self.class_names[i] = class_names[0]

        with open(self.root + '/tr_ID.txt', 'r') as fd:
            reader = csv.reader(fd)
            for i, ids in enumerate(reader):
                self.split.append(int(ids[0]))

    def _fill_label_names(self):
        for item in self.labels:
            self.label_names.append(self._normalize_text(self.class_names[item])) 
                
    def _edit_classes(self):
        for i in range(len(self.classes)):
            self.classes[i] = self._normalize_text(self.classes[i])

    def _normalize_text(self, text):
        return text.replace('_', ' ').lower()

    def _load_dataframe(self):
        
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
        elif self.datasplit == 4:
            self.data = self.data[self.data.split == 4]
        elif self.datasplit == 'train':
            self.data = self.data[(self.data.split == 2) | (self.data.split == 3) | (self.data.split == 4)]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_path = self.root + self.image_folder +  self.data['filepath'].iloc[idx]
        
        img = self.loader(img_path)
        if self.preprocess is not None:
            img = self.preprocess(img)         

        
        if self.target_transform is not None:
            label_name = self.data['label_name'].iloc[idx]
            label = self.target_transform(label_name)
        else:
            if self.label == 'name':
                label = self.data['label_name'].iloc[idx]
            elif self.label == 'number':
                label = self.data['label'].iloc[idx]
            else:
                print('Error: label-parameter: name or number')
    
        return img, label


    #root = '/home/korsch_ssd/datasets/moths/MCC/8classes'
    #preprocess = None #or CLIP Preprocess
    #templates = ["a photo of a {}."]
    #target_transform = templates[0].format
    #moth = mmc_moths(root, target_transform=target_transform, transform=preprocess, datasplit=None) # all Data
    #moth0 = mmc_moths(root, target_transform=target_transform, transform=preprocess, datasplit=0) # test_set
    #moth1 = mmc_moths(root, target_transform=target_transform, transform=preprocess, datasplit=1) # val_set
    #moth2 = mmc_moths(root, target_transform=target_transform, transform=preprocess, datasplit=2) # first train_part
    #moth3 = mmc_moths(root, target_transform=target_transform, transform=preprocess, datasplit=3) # second train_part
    #moth4 = mmc_moths(root, target_transform=target_transform, transform=preprocess, datasplit=4) # third train_part
    #test_set = mmc_moths(root, target_transform=target_transform, transform=preprocess, datasplit='test')
    #val_set = mmc_moths(root, target_transform=target_transform, transform=preprocess, datasplit='val')
    #train_set = mmc_moths(root, target_transform=target_transform, transform=preprocess, datasplit='train')