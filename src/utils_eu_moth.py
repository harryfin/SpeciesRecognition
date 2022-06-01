import torch
from datasetEuMoths.eu_moths import eu_moths


def indices_for_split(all_moths):
    train_indices = all_moths.data[all_moths.data.label < 100].index.tolist()
    val_indices = all_moths.data[(all_moths.data.label >= 100) & (
        all_moths.data.label < 150)].index.tolist()
    test_indices = all_moths.data[all_moths.data.label >= 150].index.tolist()
    return train_indices, val_indices, test_indices


def split_dataset(all_moths, train_indices, val_indices, test_indices):

    train_dataset = torch.utils.data.Subset(all_moths, train_indices)
    val_dataset = torch.utils.data.Subset(all_moths, val_indices)
    test_dataset = torch.utils.data.Subset(all_moths, test_indices)

    return train_dataset, val_dataset, test_dataset


def get_dataloader(train_dataset, val_dataset, test_dataset, batch_size):

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    return train_data_loader, val_data_loader, test_data_loader


def split_dataloader_from_dataset(all_moths, batch_size):
    """
    example call:
    > from datasetEuMoths.eu_moths import eu_moths
    > import utils_eu_moth as emu
    > import clip
    > _ , preprocess = clip.load('RN50',  device='cpu', jit=False) 
    > root_cropped = '/path/to/datasets/moths/eu_moths/cropped/ORIGINAL'
    > all_moths = eu_moths(root_cropped, transform=preprocess, datasplit=None)
    > train_d val_d,test_d = emu.split_dataloader_from_dataset(all_moths, batch_size=16)
    """

    train_indices, val_indices, test_indices = indices_for_split(all_moths)
    train_dataset, val_dataset, test_dataset = split_dataset(
        all_moths, train_indices, val_indices, test_indices)
    train_data_loader, val_data_loader, test_data_loader = get_dataloader(
        train_dataset, val_dataset, test_dataset, batch_size)
    return train_data_loader, val_data_loader, test_data_loader


# Load Data (normal split)
def load_data(args, preprocess):

    if args.label == "english":
        print("label")
        test_set = eu_moths(
            args.root,
            transform=preprocess,
            datasplit="test",
            label=args.label,
        )
        val_set = eu_moths(
            args.root,
            transform=preprocess,
            datasplit="val",
            label=args.label,
        )
        train_set = eu_moths(
            args.root,
            transform=preprocess,
            datasplit="train",
            label=args.label,
        )
    else:
        test_set = eu_moths(args.root, transform=preprocess, datasplit="test")
        val_set = eu_moths(args.root, transform=preprocess, datasplit="val")
        train_set = eu_moths(
            args.root, transform=preprocess, datasplit="train")

    return (test_set, val_set, train_set)
