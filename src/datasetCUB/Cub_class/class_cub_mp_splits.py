from datasetCUB.Cub_class.class_cub import Cub2011
from datasetCUB.Cub_class.class_cub_for_simclr import Cub2011_with_3_image_augmentation
from datasetCUB.Cub_class.class_cub_with_data_augmentation import (
    Cub2011_data_augmentation_for_training,
)
from mat4py import loadmat
from utils_cub import unpack_mat_split
import sys
import torch
import utils as uu


class class_cub_mp_split:

    def __init__(
        self,
        root,
        cub_root,
        preprocess_clip=None,
        preprocess_simclr=None,
        preprocess_training=None,
        split="PS",
        simclr_augmentation=None,
        data_augmentation=None,
    ):

        self.root = root
        self.cub_root = cub_root
        self.filename_ss = self.root + \
            "/data-set-extensions/Zero-Shot-Split-Sets/standard_split/CUB/att_splits.mat"
        self.filename_ps = self.root + \
            "/data-set-extensions/Zero-Shot-Split-Sets/xlsa17/data/CUB/att_splits.mat"

        # load Cub-Dataset
        self.preprocess_clip = preprocess_clip
        self.preprocess_simclr = preprocess_simclr
        self.preprocess_training = preprocess_training

        if simclr_augmentation is not None:
            self.cub = Cub2011(root=self.cub_root,
                               transform_image=None, train=None)
            self.dataset = Cub2011_with_3_image_augmentation(
                self.cub_root,
                self.cub.data,
                self.preprocess_clip,
                self.preprocess_simclr,
                self.cub.classes,
            )
        elif data_augmentation is not None:
            self.cub = Cub2011(root=self.cub_root,
                               transform_image=None, train=None)
            self.dataset = Cub2011_data_augmentation_for_training(
                self.cub_root,
                self.cub.data,
                self.preprocess_clip,
                self.preprocess_training,
                self.cub.classes,
            )
        else:
            self.cub = Cub2011(
                root=self.cub_root, transform_image=self.preprocess_clip, train=None
            )
            self.dataset = self.cub

        # choose split
        self.split = split
        if self.split not in ["PS", "SS"]:
            sys.exit("split not available - choose from PS or SS")

        # load data an indices
        self.trainval_loc = None
        self.test_seen_loc = None
        self.test_unseen_loc = None

        self._load_data()

        # load dataset
        self.trainval_dataset = None
        self.test_seen_dataset = None
        self.test_unseen_dataset = None

        self._load_dataset()

        # get classes

        self.train_class_SS = None
        self.train_class_PS = None
        self.test_unseen_class_SS = None
        self.test_unseen_class_PS = None
        self.test_seen_class_PS = None

        self.get_classes()

        # dataloader
        self.trainval_data_loader = None
        self.test_unseen_data_loader = None
        self.test_seen_data_loader = None

    def _load_data(self):

        if self.split == "PS":
            filename = self.filename_ps
        else:
            filename = self.filename_ss

        data = loadmat(filename)

        if self.split == "PS":
            self.trainval_loc = unpack_mat_split(data["trainval_loc"])
            self.test_seen_loc = unpack_mat_split(data["test_seen_loc"])
            self.test_unseen_loc = unpack_mat_split(data["test_unseen_loc"])
        else:
            self.trainval_loc = unpack_mat_split(data["trainval_loc"])
            self.test_unseen_loc = unpack_mat_split(data["test_unseen_loc"])

    def _load_dataset(self):

        if self.split == "PS":
            self.trainval_dataset = torch.utils.data.Subset(
                self.dataset, self.trainval_loc
            )
            self.test_unseen_dataset = torch.utils.data.Subset(
                self.dataset, self.test_unseen_loc
            )
            self.test_seen_dataset = torch.utils.data.Subset(
                self.dataset, self.test_seen_loc
            )
        else:
            self.trainval_dataset = torch.utils.data.Subset(
                self.dataset, self.trainval_loc
            )
            self.test_unseen_dataset = torch.utils.data.Subset(
                self.dataset, self.test_unseen_loc
            )

    def get_dataset(self):
        if self.split == "PS":
            return (
                self.trainval_dataset,
                self.test_unseen_dataset,
                self.test_seen_dataset,
            )
        else:
            return self.trainval_dataset, self.test_unseen_dataset, None

    def get_dataloader(self, batch_size):
        self.trainval_data_loader = torch.utils.data.DataLoader(
            self.trainval_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_unseen_data_loader = torch.utils.data.DataLoader(
            self.test_unseen_dataset, batch_size=batch_size
        )
        if self.split == "PS":
            self.test_seen_data_loader = torch.utils.data.DataLoader(
                self.test_seen_dataset, batch_size=batch_size
            )

        if self.split == "PS":
            return (
                self.trainval_data_loader,
                self.test_unseen_data_loader,
                self.test_seen_data_loader,
            )
        else:
            return self.trainval_data_loader, self.test_unseen_data_loader, None

    def get_classes(self):

        self.train_class_SS = [
            "Spotted Catbird",
            "Crested Auklet",
            "Mourning Warbler",
            "Blue headed Vireo",
            "Pine Grosbeak",
            "Green Kingfisher",
            "Brown Thrasher",
            "Cape Glossy Starling",
            "Vermilion Flycatcher",
            "Ivory Gull",
            "Yellow Warbler",
            "Caspian Tern",
            "Common Tern",
            "Downy Woodpecker",
            "White crowned Sparrow",
            "Pelagic Cormorant",
            "Anna Hummingbird",
            "American Goldfinch",
            "Gray Catbird",
            "American Redstart",
            "Prothonotary Warbler",
            "Savannah Sparrow",
            "Tennessee Warbler",
            "Parakeet Auklet",
            "Cardinal",
            "Barn Swallow",
            "Eared Grebe",
            "Cerulean Warbler",
            "Brown Pelican",
            "Prairie Warbler",
            "Yellow throated Vireo",
            "Black Tern",
            "Laysan Albatross",
            "Palm Warbler",
            "Baird Sparrow",
            "Rose breasted Grosbeak",
            "Rufous Hummingbird",
            "Fish Crow",
            "White necked Raven",
            "Evening Grosbeak",
            "Brewer Sparrow",
            "Black capped Vireo",
            "Artic Tern",
            "Rusty Blackbird",
            "Pied Kingfisher",
            "White throated Sparrow",
            "Blue winged Warbler",
            "Philadelphia Vireo",
            "Chuck will Widow",
            "Nashville Warbler",
            "Black throated Blue Warbler",
            "Tropical Kingbird",
            "Lazuli Bunting",
            "Long tailed Jaeger",
            "Pied billed Grebe",
            "Green Jay",
            "Bronzed Cowbird",
            "Dark eyed Junco",
            "Cliff Swallow",
            "Blue Jay",
            "Nelson Sharp tailed Sparrow",
            "Elegant Tern",
            "Louisiana Waterthrush",
            "Heermann Gull",
            "California Gull",
            "Black and white Warbler",
            "Whip poor Will",
            "Bay breasted Warbler",
            "Henslow Sparrow",
            "Red bellied Woodpecker",
            "Common Yellowthroat",
            "Shiny Cowbird",
            "Orange crowned Warbler",
            "Frigatebird",
            "Nighthawk",
            "Northern Fulmar",
            "Orchard Oriole",
            "American Pipit",
            "Clay colored Sparrow",
            "Lincoln Sparrow",
            "Red cockaded Woodpecker",
            "Mangrove Cuckoo",
            "Fox Sparrow",
            "Seaside Sparrow",
            "Pileated Woodpecker",
            "Sage Thrasher",
            "Brown Creeper",
            "House Sparrow",
            "Bobolink",
            "Gray Kingbird",
            "Magnolia Warbler",
            "Scarlet Tanager",
            "Bewick Wren",
            "Rock Wren",
            "White breasted Nuthatch",
            "Clark Nutcracker",
            "Winter Wren",
            "Ruby throated Hummingbird",
            "Le Conte Sparrow",
            "Harris Sparrow",
            "Red winged Blackbird",
            "Scissor tailed Flycatcher",
            "Yellow breasted Chat",
            "Geococcyx",
            "Slaty backed Gull",
            "Red eyed Vireo",
            "Ovenbird",
            "Gadwall",
            "Mallard",
            "Kentucky Warbler",
            "Eastern Towhee",
            "House Wren",
            "Green Violetear",
            "Olive sided Flycatcher",
            "Hooded Warbler",
            "Sooty Albatross",
            "Painted Bunting",
            "Canada Warbler",
            "Song Sparrow",
            "Western Meadowlark",
            "Green tailed Towhee",
            "Cactus Wren",
            "Horned Puffin",
            "Yellow headed Blackbird",
            "Ring billed Gull",
            "Red faced Cormorant",
            "European Goldfinch",
            "Warbling Vireo",
            "Summer Tanager",
            "Herring Gull",
            "Red headed Woodpecker",
            "Least Flycatcher",
            "Glaucous winged Gull",
            "Chipping Sparrow",
            "Carolina Wren",
            "Florida Jay",
            "Hooded Merganser",
            "Forsters Tern",
            "Pine Warbler",
            "Loggerhead Shrike",
            "Blue Grosbeak",
            "Pigeon Guillemot",
            "Vesper Sparrow",
            "Worm eating Warbler",
            "Common Raven",
            "Swainson Warbler",
            "Ringed Kingfisher",
            "Red breasted Merganser",
            "Myrtle Warbler",
            "Horned Lark",
        ]
        self.train_class_PS = [
            "Spotted Catbird",
            "Crested Auklet",
            "Mourning Warbler",
            "Blue headed Vireo",
            "Pine Grosbeak",
            "Green Kingfisher",
            "Brown Thrasher",
            "Cape Glossy Starling",
            "Vermilion Flycatcher",
            "Ivory Gull",
            "Yellow Warbler",
            "Common Tern",
            "Downy Woodpecker",
            "Belted Kingfisher",
            "Pelagic Cormorant",
            "Anna Hummingbird",
            "Great Crested Flycatcher",
            "American Goldfinch",
            "Pacific Loon",
            "Gray Catbird",
            "American Redstart",
            "Prothonotary Warbler",
            "Tennessee Warbler",
            "Parakeet Auklet",
            "Cardinal",
            "Eared Grebe",
            "Brown Pelican",
            "Prairie Warbler",
            "Black Tern",
            "Western Gull",
            "Laysan Albatross",
            "Palm Warbler",
            "Rose breasted Grosbeak",
            "Horned Grebe",
            "Rufous Hummingbird",
            "Fish Crow",
            "White necked Raven",
            "White breasted Kingfisher",
            "Brewer Sparrow",
            "Black capped Vireo",
            "White Pelican",
            "Artic Tern",
            "Western Wood Pewee",
            "Least Tern",
            "Rusty Blackbird",
            "Pied Kingfisher",
            "White throated Sparrow",
            "Philadelphia Vireo",
            "Chuck will Widow",
            "Nashville Warbler",
            "Baltimore Oriole",
            "Black throated Blue Warbler",
            "Lazuli Bunting",
            "Long tailed Jaeger",
            "Northern Waterthrush",
            "Green Jay",
            "Dark eyed Junco",
            "Cliff Swallow",
            "Blue Jay",
            "Nelson Sharp tailed Sparrow",
            "Bohemian Waxwing",
            "Elegant Tern",
            "Louisiana Waterthrush",
            "Heermann Gull",
            "California Gull",
            "Black and white Warbler",
            "American Three toed Woodpecker",
            "Least Auklet",
            "Whip poor Will",
            "Marsh Wren",
            "Bay breasted Warbler",
            "Bank Swallow",
            "Red bellied Woodpecker",
            "Shiny Cowbird",
            "Frigatebird",
            "Nighthawk",
            "Brewer Blackbird",
            "Clay colored Sparrow",
            "Lincoln Sparrow",
            "Mangrove Cuckoo",
            "Fox Sparrow",
            "Seaside Sparrow",
            "Sage Thrasher",
            "Golden winged Warbler",
            "House Sparrow",
            "Bobolink",
            "Western Grebe",
            "Gray Kingbird",
            "Grasshopper Sparrow",
            "Great Grey Shrike",
            "Bewick Wren",
            "Rock Wren",
            "Clark Nutcracker",
            "Winter Wren",
            "Black footed Albatross",
            "Ruby throated Hummingbird",
            "Harris Sparrow",
            "Red winged Blackbird",
            "Gray crowned Rosy Finch",
            "Scissor tailed Flycatcher",
            "Geococcyx",
            "Yellow breasted Chat",
            "Red eyed Vireo",
            "Slaty backed Gull",
            "Ovenbird",
            "Gadwall",
            "Eastern Towhee",
            "House Wren",
            "Olive sided Flycatcher",
            "Hooded Warbler",
            "Sooty Albatross",
            "Painted Bunting",
            "Hooded Oriole",
            "Purple Finch",
            "Canada Warbler",
            "Indigo Bunting",
            "Song Sparrow",
            "Western Meadowlark",
            "Cactus Wren",
            "American Crow",
            "Horned Puffin",
            "Cedar Waxwing",
            "Ring billed Gull",
            "European Goldfinch",
            "Red faced Cormorant",
            "Warbling Vireo",
            "Summer Tanager",
            "Herring Gull",
            "Northern Flicker",
            "Carolina Wren",
            "Least Flycatcher",
            "Chipping Sparrow",
            "Glaucous winged Gull",
            "Florida Jay",
            "Hooded Merganser",
            "Forsters Tern",
            "Pine Warbler",
            "Acadian Flycatcher",
            "Blue Grosbeak",
            "Pigeon Guillemot",
            "Black throated Sparrow",
            "Worm eating Warbler",
            "Vesper Sparrow",
            "Common Raven",
            "Swainson Warbler",
            "Ringed Kingfisher",
            "Red breasted Merganser",
            "Rhinoceros Auklet",
            "Myrtle Warbler",
            "Horned Lark",
        ]

        self.test_unseen_class_SS = [
            "Brandt Cormorant",
            "Brewer Blackbird",
            "White breasted Kingfisher",
            "Boat tailed Grackle",
            "Black billed Cuckoo",
            "White Pelican",
            "Groove billed Ani",
            "Western Wood Pewee",
            "Least Tern",
            "Tree Swallow",
            "Golden winged Warbler",
            "Hooded Oriole",
            "Purple Finch",
            "Red legged Kittiwake",
            "Indigo Bunting",
            "Pomarine Jaeger",
            "White eyed Vireo",
            "Belted Kingfisher",
            "Western Grebe",
            "American Crow",
            "Great Crested Flycatcher",
            "Grasshopper Sparrow",
            "Baltimore Oriole",
            "Yellow bellied Flycatcher",
            "Great Grey Shrike",
            "Pacific Loon",
            "Scott Oriole",
            "Cedar Waxwing",
            "Northern Waterthrush",
            "Field Sparrow",
            "Northern Flicker",
            "Cape May Warbler",
            "Bohemian Waxwing",
            "Yellow billed Cuckoo",
            "Chestnut sided Warbler",
            "Black footed Albatross",
            "Acadian Flycatcher",
            "Tree Sparrow",
            "Mockingbird",
            "American Three toed Woodpecker",
            "Western Gull",
            "Least Auklet",
            "Marsh Wren",
            "Black throated Sparrow",
            "Gray crowned Rosy Finch",
            "Sayornis",
            "Bank Swallow",
            "Wilson Warbler",
            "Rhinoceros Auklet",
            "Horned Grebe",
        ]
        self.test_unseen_class_PS = [
            "Brandt Cormorant",
            "Mallard",
            "Kentucky Warbler",
            "Northern Fulmar",
            "Evening Grosbeak",
            "Orchard Oriole",
            "American Pipit",
            "Boat tailed Grackle",
            "Black billed Cuckoo",
            "Green Violetear",
            "Red cockaded Woodpecker",
            "Groove billed Ani",
            "Caspian Tern",
            "Pileated Woodpecker",
            "Tree Swallow",
            "Blue winged Warbler",
            "Red legged Kittiwake",
            "Brown Creeper",
            "White crowned Sparrow",
            "Pomarine Jaeger",
            "White eyed Vireo",
            "Green tailed Towhee",
            "Yellow bellied Flycatcher",
            "Yellow headed Blackbird",
            "Tropical Kingbird",
            "Scott Oriole",
            "Magnolia Warbler",
            "Scarlet Tanager",
            "Field Sparrow",
            "Pied billed Grebe",
            "Bronzed Cowbird",
            "Red headed Woodpecker",
            "White breasted Nuthatch",
            "Savannah Sparrow",
            "Cape May Warbler",
            "Yellow billed Cuckoo",
            "Chestnut sided Warbler",
            "Loggerhead Shrike",
            "Barn Swallow",
            "Cerulean Warbler",
            "Tree Sparrow",
            "Mockingbird",
            "Yellow throated Vireo",
            "Le Conte Sparrow",
            "Sayornis",
            "Henslow Sparrow",
            "Wilson Warbler",
            "Baird Sparrow",
            "Common Yellowthroat",
            "Orange crowned Warbler",
        ]
        self.test_seen_class_PS = [
            "Spotted Catbird",
            "Crested Auklet",
            "Mourning Warbler",
            "Blue headed Vireo",
            "Pine Grosbeak",
            "Green Kingfisher",
            "Brown Thrasher",
            "Cape Glossy Starling",
            "Vermilion Flycatcher",
            "Ivory Gull",
            "Yellow Warbler",
            "Common Tern",
            "Downy Woodpecker",
            "Belted Kingfisher",
            "Pelagic Cormorant",
            "Great Crested Flycatcher",
            "Anna Hummingbird",
            "American Goldfinch",
            "Pacific Loon",
            "Gray Catbird",
            "American Redstart",
            "Prothonotary Warbler",
            "Tennessee Warbler",
            "Parakeet Auklet",
            "Cardinal",
            "Eared Grebe",
            "Brown Pelican",
            "Prairie Warbler",
            "Black Tern",
            "Western Gull",
            "Laysan Albatross",
            "Palm Warbler",
            "Rose breasted Grosbeak",
            "Horned Grebe",
            "Rufous Hummingbird",
            "Fish Crow",
            "White necked Raven",
            "White breasted Kingfisher",
            "Brewer Sparrow",
            "Black capped Vireo",
            "White Pelican",
            "Artic Tern",
            "Western Wood Pewee",
            "Least Tern",
            "Rusty Blackbird",
            "Pied Kingfisher",
            "White throated Sparrow",
            "Philadelphia Vireo",
            "Chuck will Widow",
            "Nashville Warbler",
            "Baltimore Oriole",
            "Black throated Blue Warbler",
            "Lazuli Bunting",
            "Long tailed Jaeger",
            "Northern Waterthrush",
            "Green Jay",
            "Dark eyed Junco",
            "Cliff Swallow",
            "Blue Jay",
            "Nelson Sharp tailed Sparrow",
            "Elegant Tern",
            "Bohemian Waxwing",
            "Louisiana Waterthrush",
            "Heermann Gull",
            "California Gull",
            "Black and white Warbler",
            "American Three toed Woodpecker",
            "Least Auklet",
            "Whip poor Will",
            "Marsh Wren",
            "Bay breasted Warbler",
            "Bank Swallow",
            "Red bellied Woodpecker",
            "Shiny Cowbird",
            "Frigatebird",
            "Nighthawk",
            "Brewer Blackbird",
            "Clay colored Sparrow",
            "Lincoln Sparrow",
            "Mangrove Cuckoo",
            "Fox Sparrow",
            "Seaside Sparrow",
            "Sage Thrasher",
            "Golden winged Warbler",
            "House Sparrow",
            "Bobolink",
            "Western Grebe",
            "Gray Kingbird",
            "Grasshopper Sparrow",
            "Great Grey Shrike",
            "Bewick Wren",
            "Rock Wren",
            "Clark Nutcracker",
            "Winter Wren",
            "Black footed Albatross",
            "Ruby throated Hummingbird",
            "Harris Sparrow",
            "Red winged Blackbird",
            "Gray crowned Rosy Finch",
            "Scissor tailed Flycatcher",
            "Geococcyx",
            "Yellow breasted Chat",
            "Red eyed Vireo",
            "Slaty backed Gull",
            "Ovenbird",
            "Gadwall",
            "Eastern Towhee",
            "House Wren",
            "Olive sided Flycatcher",
            "Hooded Warbler",
            "Sooty Albatross",
            "Painted Bunting",
            "Hooded Oriole",
            "Purple Finch",
            "Canada Warbler",
            "Indigo Bunting",
            "Song Sparrow",
            "Western Meadowlark",
            "Cactus Wren",
            "American Crow",
            "Horned Puffin",
            "Cedar Waxwing",
            "Ring billed Gull",
            "European Goldfinch",
            "Warbling Vireo",
            "Red faced Cormorant",
            "Summer Tanager",
            "Herring Gull",
            "Northern Flicker",
            "Carolina Wren",
            "Least Flycatcher",
            "Glaucous winged Gull",
            "Chipping Sparrow",
            "Florida Jay",
            "Pine Warbler",
            "Forsters Tern",
            "Hooded Merganser",
            "Acadian Flycatcher",
            "Blue Grosbeak",
            "Pigeon Guillemot",
            "Vesper Sparrow",
            "Black throated Sparrow",
            "Worm eating Warbler",
            "Common Raven",
            "Swainson Warbler",
            "Ringed Kingfisher",
            "Red breasted Merganser",
            "Rhinoceros Auklet",
            "Myrtle Warbler",
            "Horned Lark",
        ]


# example Call

# split = 'SS'
# dataset_class = class_cub_mp_split(split)
# trainval_dataset, test_unseen_dataset = dataset_class.get_dataset()
# trainval_data_loader, test_unseen_data_loader = dataset_class.get_dataloader(batch_size)
# split = 'PS'
# dataset_class = class_cub_mp_split(split)
# trainval_dataset, test_unseen_dataset, test_seen_dataset = dataset_class.get_dataset()
# trainval_data_loader, test_unseen_data_loader, test_seen_data_loader = dataset_class.get_dataloader(batch_size)
