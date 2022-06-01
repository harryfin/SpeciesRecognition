import random
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import (
    Compose,
    Resize,
    RandomCrop,
    CenterCrop,
    ToTensor,
    Normalize,
    ColorJitter,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomApply,
    RandomGrayscale,
)
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def preprocess_with_data_augmentation_for_simclr_training_slip(model):
    return _transform_3(model.visual.input_resolution)


def preprocess_with_data_augmentation_for_simclr_training(model):
    return _transform_2(model.visual.input_resolution)


def preprocess_with_data_augmentation_for_clip_training(model):
    return _transform(model.visual.input_resolution)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            RandomHorizontalFlip(),
            RandomResizedCrop(n_px, scale=(0.5, 1.0)),
            # CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def _transform_2(n_px, s=1.0):
    # transformation from MoCo v3 (https://arxiv.org/pdf/2002.05709.pdf)
    #   rnd_color_jitter
    #   nd_gray
    # remaining transformation for to fit CLIP Model / Dataloader

    color_jitter = ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = RandomApply([color_jitter], p=0.8)
    rnd_gray = RandomGrayscale(p=0.2)

    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            rnd_color_jitter,
            rnd_gray,
            CenterCrop(n_px),  # or RandomCrop
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class _GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709 - Facebook-Implemention (SLIP)"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def _transform_3(n_px):
    """DataAugmentation for SimCLR- Facebook-Implemention (SLIP)"""

    # normalization for Facebook SLIP implementation
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(n_px, scale=(0.08, 1.0)),
            transforms.RandomApply(
                # not strengthened
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([_GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            _convert_image_to_rgb,  # add that because of dimension failure
            transforms.ToTensor(),
            # normalize,
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
