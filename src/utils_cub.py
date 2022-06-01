import torch
import wandb
import clip
import numpy as np
import utils as ut
import os.path
from datasetCUB.Cub_class.class_cub import Cub2011
from sklearn.model_selection import train_test_split
from datasetCUB.Cub_class.class_cub import Cub2011


def _load_dataset_with_split(cub, split_classes):

    filter_list = _load_splits(split_classes)
    df_set = cub.data.query('img_class == @filter_list')

    return df_set


def _load_splits(root, split_classes):

    root_split = os.path.join(
        root, 'data-set-extensions/Zero-Shot-Split-Sets/xlsa17/data/CUB/')

    choice = [
        'allclasses',
        'testclasses',
        'trainclasses1',
        'trainclasses2',
        'trainclasses3',
        'trainvalclasses',
        'valclasses1',
        'valclasses2',
        'valclasses3',
    ]

    if split_classes in choice:

        with open(root_split + split_classes + '.txt') as f:
            lines = f.readlines()

        for ind, line in enumerate(lines):
            lines[ind] = line.replace('\n', '')
            lines[ind], _ = line.split('.')
            lines[ind] = int(lines[ind])

        return lines
    else:
        print('No correct split_class - choose from', choice)
        return None


def _get_full_cub(root):
    root_cub = ut.get_root_CUB(root)
    cub = Cub2011(root=root_cub, train=None)
    return cub


def _argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def rescale_img(image):
    # rescale Image
    return (image - image.min()) / (image.max() - image.min())


def load_zero_shot_subsets(dataset, zs_split_indicies):
    pass


def unpack_mat_split(list_in):
    list_out = []
    for i in list_in:
        list_out.append(i[0] - 1)
    return list_out


def save_model(model, PATH):
    torch.save(model, PATH)


def load_model(PATH):
    # Model class must be defined somewhere
    model = torch.load(PATH)
    model.eval()
    return model


def write_logfile(epoch, train_acc, val_acc):
    f = open("log_{}.txt".format(epoch), "a")
    f.write("epoch: {}".format(epoch))
    f.write("train accuracy: {}".format(train_acc))
    f.write("val accuracy: {}".format(val_acc))
    f.close()


def log_dict(name, dict, list):
    # name: epoch or index or something
    # list
    f = open("log_{}.txt".format(name), "a")

    for i in list:
        if i in dict.keys():
            f.write(str(i) + ": {}".format(dict[i]))
        else:
            f.write(str(i) + "can not be logged")
    f.close()


def get_default_params(model_name):
    # Params from paper https://arxiv.org/pdf/2103.00020.pdf Page 48
    if model_name in ["RN50", "RN101", "RN50x4"]:
        return {
            "lr": 5.0e-4,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1.0e-8,
            "weight_decay": 0.2,
        }
    elif model_name == "ViT-B/32":
        return {
            "lr": 5.0e-4,
            "beta1": 0.9,
            "beta2": 0.98,
            "eps": 1.0e-6,
            "weight_decay": 0.2,
        }
    else:
        return {}


def log_wandb(logs, logging_list):
    wandb_log = {}
    for log_arg in logging_list:
        wandb_log[log_arg] = logs[log_arg]

    wandb.log(wandb_log)


def get_device(gpu=0):
    device = "cuda:" + str(gpu) if torch.cuda.is_available() else "cpu"
    return device


def get_dataloader_from_dataset(cub_train_val, cub_test, validation_split, batch_size):
    # https://linuxtut.com/en/c6023453e00bfead9e9f/
    train_indices, val_indices = train_test_split(
        list(range(len(cub_train_val.data["img_class"].to_numpy()))),
        test_size=validation_split,
        stratify=cub_train_val.data["img_class"].to_numpy(),
    )
    train_dataset = torch.utils.data.Subset(cub_train_val, train_indices)
    val_dataset = torch.utils.data.Subset(cub_train_val, val_indices)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        cub_test, batch_size=batch_size)

    return train_data_loader, val_data_loader, test_data_loader


def get_dataloader_from_dataset_with_different_preprocesses(
    cub_train_val_preprocess_train,
    cub_train_val_preprocess_val,
    cub_test,
    validation_split,
    batch_size,
):

    train_indices, val_indices = train_test_split(
        list(
            range(len(cub_train_val_preprocess_train.data["img_class"].to_numpy()))),
        test_size=validation_split,
        stratify=cub_train_val_preprocess_train.data["img_class"].to_numpy(),
    )
    train_dataset = torch.utils.data.Subset(
        cub_train_val_preprocess_train, train_indices
    )
    val_dataset = torch.utils.data.Subset(
        cub_train_val_preprocess_val, val_indices)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        cub_test, batch_size=batch_size)

    return train_data_loader, val_data_loader, test_data_loader


def load_clip(device, model_architecture, seed=None, cuda_det=None):

    model, preprocess = clip.load(
        model_architecture, device=device, jit=False
    )  # Must set jit=False for training

    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(
            model
        )
    if seed is not None:
        torch.manual_seed(seed)
    if cuda_det is not None:
        torch.backends.cudnn.deterministic = cuda_det
    return model, preprocess


def get_CUB_dataloader(root, batch_size, validation_split, preprocess):
    cub_root = root_cub = ut.get_root_CUB(root)
    cub_train_val = Cub2011(
        root=cub_root, train=True, transform_image=preprocess, label_mapping=False
    )
    cub_test = Cub2011(
        root=cub_root, train=False, transform_image=preprocess, label_mapping=False
    )
    train_data_loader, val_data_loader, test_data_loader = get_dataloader_from_dataset(
        cub_train_val, cub_test, validation_split, batch_size
    )

    return train_data_loader, val_data_loader, test_data_loader


def get_dataloader_from_dataset(cub_train_val, cub_test, validation_split, batch_size):
    # https://linuxtut.com/en/c6023453e00bfead9e9f/
    train_indices, val_indices = train_test_split(
        list(range(len(cub_train_val.data["img_class"].to_numpy()))),
        test_size=validation_split,
        stratify=cub_train_val.data["img_class"].to_numpy(),
    )
    train_dataset = torch.utils.data.Subset(cub_train_val, train_indices)
    val_dataset = torch.utils.data.Subset(cub_train_val, val_indices)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        cub_test, batch_size=batch_size)

    return train_data_loader, val_data_loader, test_data_loader


def save_features(name, data):
    np.save(name, data)


def load_features(name):
    return np.load(name)


# Load Data
def load_data(root, preprocess):
    cub_root = ut.get_root_CUB(root)

    cub_train_val = Cub2011(
        root=cub_root, transform_image=preprocess, train=True)
    cub_test = Cub2011(root=cub_root, transform_image=preprocess, train=False)

    return cub_train_val, cub_test
