import torch
import clip


def _get_normalized_CLIP_features(list_image, list_txt, model, device):

    images = torch.stack(
        [img for img in list_image], dim=0
    )  # changed how it said above

    image_features = model.encode_image(images.to(device))
    text_features = model.encode_text(list_txt.to(device))

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return (image_features, text_features)


def _get_normalized_CLIP_features_for_img_img(
    list_image_1, list_image_2, model, device
):

    images1 = torch.stack(
        [img for img in list_image_1], dim=0
    )  # changed how it said above

    images2 = torch.stack(
        [img for img in list_image_2], dim=0
    )  # changed how it said above

    image_features1 = model.encode_image(images1.to(device))
    image_features2 = model.encode_image(images2.to(device))

    # normalized features
    image_features1 = image_features1 / \
        image_features1.norm(dim=-1, keepdim=True)
    image_features2 = image_features2 / \
        image_features2.norm(dim=-1, keepdim=True)
    return (image_features1, image_features2)


def _get_CLIP_logits(image_features, text_features, model):

    # cosine similarity as logits
    # set fix logit_scale: https://github.com/openai/CLIP/issues/46
    logit_scale = model.logit_scale.exp()
    # logit_scale = nn.Parameter(torch.tensor([np.log(1/0.07)])).half().to(self.device)
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()

    return logits_per_image, logits_per_text


def load_model(device, model_architectur):
    # load model
    model, preprocess = clip.load(
        model_architectur, device=device, jit=False
    )  # Must set jit=False for training
    if device == "cpu":
        model.float()
    else:
        model.cuda().eval()  # testen
        clip.model.convert_weights(
            model
        )  # Actually this line is unnecessary since clip by default already on float16
    return model, preprocess
