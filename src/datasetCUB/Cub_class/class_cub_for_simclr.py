from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets.folder import default_loader
import os


class Cub2011_with_3_image_augmentation(Dataset):

    base_folder = "CUB_200_2011/images"

    def __init__(self, root, cub_data, preprocess_clip, preprocess_simclr, classes):
        '''
        cub_data: pandas dataframe with cub images and labels
        '''
        self.root = root
        self.img_filepaths = cub_data['filepath']
        self.labels = cub_data['img_class']
        self.preprocess_clip = preprocess_clip
        self.preprocess_simclr = preprocess_simclr
        self.classes = classes

        print('a label looks like: ', self.__getitem__(0)[0][1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        label = self.classes[self.labels[idx] - 1]
        img_path = os.path.join(self.root, self.base_folder, self.img_filepaths[idx])
        #import pdb; pdb.set_trace()
        img = Image.open(img_path)

        img_clip = self.preprocess_clip(img)
        img_simclr1 = self.preprocess_simclr(img)
        img_simclr2 = self.preprocess_simclr(img)

        return (img_clip, label), (img_simclr1, img_simclr2)