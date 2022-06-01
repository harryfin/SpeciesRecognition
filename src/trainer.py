# https://github.com/openai/CLIP/issues/83
# https://docs.chainer.org/en/stable/glance.html

# https://github.com/KeremTurgutlu/self_supervised/blob/fastai_update/examples/training_moco_iwang.ipynb

import clip
import wandb
import copy
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers.optimization import get_scheduler
from loss_functions import CLIPLoss, SIMCLRLoss
from utils_clip import (
    _get_normalized_CLIP_features,
    _get_normalized_CLIP_features_for_img_img,
)
from weight_space_ensembling import weight_space_ensembling as wse


class Trainer:
    def __init__(
        self,
        model,
        config,  # Dictonary with config parameter
        device,  # cuda device
        classes=None,  # All classes
        wandb_log=False,  # online logging with weights and biases
        transform_label=None,  # Promptengineering -
        training=True,  # False: testing mode (some parameter must not be set)
        trainclasses=None,  # Searchspace for training accuracy
        valclasses=None,  # Searchspace for validation accuracy
        testclasses=None,  # Searchspace for testing accuracy
        # Searchspace for testing accuracy for seen classes during training
        test_seen_classes=None,
        # Searchspace for testing accuracy for unseen classes during training
        test_unseen_classes=None,
        wse=False,  # weight space ensambling
        robust_loss_MSE=False,  # add robust-loss
        simclr=False,  # add simclr-loss
        training_data_augmentation=False,  # add data_augmentation for training
        clip_scale=1,
        simclr_scale=1,
        robust_scale=1,
    ):
        self.model = model
        self.wandb_log = wandb_log
        self.classes = classes
        self.trainclasses = trainclasses
        self.valclasses = valclasses
        self.testclasses = testclasses
        self.test_seen_classes = test_seen_classes
        self.test_unseen_classes = test_unseen_classes

        self.device = device
        self.transform_label = transform_label
        self.training = training
        self.wse = wse
        self.robust_loss_MSE = robust_loss_MSE
        self.simclr = simclr
        self.training_data_augmentation = training_data_augmentation
        self.clip_scale = clip_scale
        self.simclr_scale = simclr_scale
        self.robust_scale = robust_scale

        if self.wse or self.robust_loss_MSE:
            self.model_untuned = copy.deepcopy(model)

        self.trans = transforms.ToPILImage()

        self.valset_size = None
        self.trainset_size = None
        self.testset_size = None
        self.val_dataloader = None
        self.train_dataloader = None
        self.test_dataloader = None

        self.data_loaded = False

        self.logs = {}
        self._init_logs()

        # Wandb Config
        self.config = config
        try:
            self.mode = self.config.mode
        except:
            self.mode = "train"

        self.clip_loss = CLIPLoss()
        self.robust_loss = nn.MSELoss(
            reduction="sum"
        )
        self.simclr_loss = SIMCLRLoss(self.device)

        # Model parameter
        self.betas = None
        self.optimizer = None
        self.loss_img = None
        self.loss_txt = None

        if self.training:
            self.betas = (self.config.beta1, self.config.beta2)
            self.set_optimizer()

        self.scheduler = None

        # additonal parameter setting
        self.log_rescale = True

        # check if classes are set correctly
        self.check_classes()

    def check_classes(self):
        "Normally train, val and testclasses are identical. If not set it explicitily"
        if self.trainclasses == None:
            self.trainclasses = self.classes if self.classes is not None else print(
                'Information: no trainclasses is set')
        if self.valclasses == None:
            self.valclasses = self.classes if self.classes is not None else print(
                'Information: no valclasses is set')
        if self.testclasses == None:
            self.testclasses = self.classes if self.classes is not None else print(
                'Information: no testclasses (normal testsplit (not the zero-shot-setting testsplit)) is set ')

    def set_scheduler(self):
        if self.train_dataloader is not None:
            self.scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=self.config.warmup_steps *
                len(self.train_dataloader),
                num_training_steps=self.config.epochs *
                len(self.train_dataloader),
            )
        else:
            print("Error: first set training Dataloder before setting scheduler")

    def run_train_test(self, validation=True):
        # standard run with logging

        self.val()
        self._log(["Val Loss", "Val Accuracy"])

        for epoch in tqdm(range(1, self.config.epochs + 1)):
            self.train()

            if validation:
                self.val()
                self._log(
                    [
                        "Train Loss",
                        "Train Accuracy",
                        "Val Accuracy",
                        # "val_guess",
                        # "val_truth",
                        "Val Loss",
                    ]
                )
            else:
                self._log(["Train Loss", "Train Accuracy"])
        self.test()
        self._log(["Test Accuracy", "Test Loss"])

    def set_data_loader(
        self, train_data_loader=None,
        val_data_loader=None,
        test_data_loader=None,
        test_seen_data_loader=None,
        test_unseen_data_loader=None
    ):

        if train_data_loader is not None:
            self.train_dataloader = train_data_loader
            self.trainset_size = len(train_data_loader.dataset)

        if val_data_loader is not None:
            self.val_dataloader = val_data_loader
            self.valset_size = len(val_data_loader.dataset)

        if test_data_loader is not None:
            self.test_dataloader = test_data_loader
            self.testset_size = len(test_data_loader.dataset)

        if test_seen_data_loader is not None:
            self.test_seen_dataloader = test_seen_data_loader
            self.testset_seen_size = len(test_seen_data_loader.dataset)

        if test_unseen_data_loader is not None:
            self.test_unseen_dataloader = test_unseen_data_loader
            self.testset_unseen_size = len(test_unseen_data_loader.dataset)

    def set_optimizer(self):

        if self.config.optim == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                betas=self.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )

        elif self.config.optim == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                betas=self.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            )
        else:
            print("error: optim unkown\nchoice: Adam / AdamW\nset defaut to AdamW")
            self.config.optim = "AdamW"
            self.set_optimizer()

    def train(self):
        # model.train() # keep model in evaluation mode

        self._clear_train_logs()

        for i, batch in enumerate(self.train_dataloader):

            self.training_step(i, batch)
            self._train_acc(batch)

        # scale Loss-Logs
        self.logs["clip_loss"] /= self.trainset_size
        self.logs["robust_loss"] /= self.trainset_size
        self.logs["simclr_loss"] /= self.trainset_size

        self.logs["Train Loss"] /= self.trainset_size
        self.logs["Train Accuracy"] = sum(self.logs["train_correct"]) / len(
            self.logs["train_correct"]
        )

        print("Train Accuracy is : " + str(self.logs["Train Accuracy"]))

    def training_step(self, i, batch):

        if self.mode == "train":
            self.model.train()
        else:
            self.model.eval()

        self.optimizer.zero_grad()

        if self.simclr:
            clip_batch, simclar_batch = batch
            list_image, list_txt = clip_batch
            list_imgs_aug_1, list_imgs_aug_2 = simclar_batch
        elif self.training_data_augmentation:
            clip_batch, augmentation_batch = batch
            _, _ = clip_batch
            list_image, list_txt = augmentation_batch
        else:
            list_image, list_txt = batch

        # Features
        text_inputs = self.set_text_inputs(label=list_txt, transform=True)
        self.logs["Train Prompt Input for training"] += self.log_promt(
            label=list_txt, transform=True
        )

        # CLIP Features
        image_features, text_features = _get_normalized_CLIP_features(
            list_image, text_inputs, self.model, self.device
        )
        # CLIP Logits
        logits_per_image = (
            self.model.logit_scale.exp() * image_features @ text_features.t()
        )
        logits_per_text = (
            self.model.logit_scale.exp() * text_features @ image_features.t()
        )

        # CLIP LOSS
        clip_loss = self.clip_loss(logits_per_image, logits_per_text)
        self.logs["clip_loss"] += clip_loss
        total_loss = clip_loss * self.clip_scale

        if self.robust_loss_MSE:

            image_features_robust, text_features_robust = _get_normalized_CLIP_features(
                list_image, text_inputs, self.model_untuned, self.device
            )
            (
                image_features_unrobust,
                text_features_unrobust,
            ) = _get_normalized_CLIP_features(
                list_image, text_inputs, self.model, self.device
            )

            # For MSE Loss
            robust_loss = self.robust_loss(
                image_features_unrobust, image_features_robust
            )

            self.logs["robust_loss"] += robust_loss
            total_loss += robust_loss * self.robust_scale

        if self.simclr:
            (
                image1_features_simclr,
                image2_features_simclr,
            ) = _get_normalized_CLIP_features_for_img_img(
                list_imgs_aug_1, list_imgs_aug_2, self.model, self.device
            )
            simclr_loss = self.simclr_loss(
                image1_features_simclr, image2_features_simclr
            )
            self.logs["simclr_loss"] += simclr_loss
            total_loss += simclr_loss * self.simclr_scale

        self.logs["Train Loss"] += total_loss
        total_loss.backward()

        # Opimizer
        if self.device == "cpu":
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        else:
            self._convert_models_to_fp32()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            clip.model.convert_weights(self.model)
            if self.log_rescale:
                self.model.logit_scale.data = torch.clamp(
                    self.model.logit_scale.data, 0, 4.6052
                )  # try

    def _train_acc(self, batch):
        self.model.eval()

        if self.simclr:
            clip_batch, simclar_batch = batch
            list_image, list_txt = clip_batch
            list_imgs_aug_1, list_imgs_aug_2 = simclar_batch
        elif self.training_data_augmentation:
            clip_batch, augmentation_batch = batch
            list_image, list_txt = clip_batch
            _, _ = augmentation_batch
        else:
            list_image, list_txt = batch

        with torch.no_grad():
            text_inputs = self.set_text_inputs(
                label=self.trainclasses, transform=True)

            logits_per_image, logits_per_text = self.model(
                list_image.to(self.device), text_inputs.to(self.device)
            )

            guess = self._get_guess(
                logits_per_image, classes=self.trainclasses)

        self.logs["train_guess"] += guess
        self.logs["train_truth"] += list_txt

        for i in range(len(list_txt)):
            if guess[i] == list_txt[i]:
                self.logs["train_correct"].append(1)
            else:
                self.logs["train_correct"].append(0)

    def test(self, mode='standard'):
        if mode == 'standard':
            mode = 'test'
            dataloader = self.test_dataloader
        elif mode == 'unseen':
            mode = 'test_unseen'
            dataloader = self.test_unseen_dataloader
        elif mode == 'seen':
            mode = 'test_seen'
            dataloader = self.test_seen_dataloader
        else:
            print('Error: choose: standard, unseen or seen for testing')

        self._test_accuracy(mode=mode, dataloader=dataloader)

    def val(self):
        self._test_accuracy(mode="val", dataloader=self.val_dataloader)

    def _test_accuracy(self, mode, dataloader=None):
        # mode: 'test', 'test_seen', 'test_unseen' or 'val'

        # check accuracy
        # correct if class in prompt:
        # example: "bird" in "a picture of a bird"

        if mode == "val":
            self._clear_val_logs()
            if dataloader == None:
                dataloader = self.val_dataloader
            classes = self.valclasses
            size_of_tested_set = self.valset_size

        elif mode == "test":
            self._clear_test_logs("test")
            if dataloader == None:
                dataloader = self.test_dataloader
            classes = self.testclasses
            size_of_tested_set = self.testset_size

        elif mode == "test_seen":
            self._clear_test_logs("test_seen")
            if dataloader == None:
                dataloader = self.test_seen_dataloader
            classes = self.test_seen_classes
            size_of_tested_set = self.testset_seen_size

        elif mode == "test_unseen":
            self._clear_test_logs("test_unseen")
            if dataloader == None:
                dataloader = self.test_unseen_dataloader
            classes = self.test_unseen_classes
            size_of_tested_set = self.testset_unseen_size
        else:
            sys.exit('Something went wrong with the testmode')

        log_truth = mode + "_truth"
        log_img = mode + "_imgs"
        log_guess = mode + "_guess"
        log_correct = mode + "_correct"
        log_loss = mode.capitalize() + " Loss"
        log_acc = mode.capitalize() + " Accuracy"

        self.model.eval()

        if dataloader is None:
            print("Fail in Testing Accuracy: No Dataloader available. Testing in", mode)
        else:

            for i, batch in enumerate(dataloader):

                if self.simclr:
                    clip_batch, simclar_batch = batch
                    img, img_class = clip_batch
                    list_imgs_aug_1, list_imgs_aug_2 = simclar_batch
                elif self.training_data_augmentation:
                    clip_batch, augmentation_batch = batch
                    img, img_class = clip_batch
                    _, _ = augmentation_batch
                else:
                    img, img_class = batch

                if self.wandb_log:
                    self.logs[log_img] += [
                        wandb.Image(
                            self.trans(self._rs(x)).resize(
                                (25, 25), Image.ANTIALIAS)
                        )
                        for x in img
                    ]

                with torch.no_grad():

                    text_inputs = self.set_text_inputs(
                        label=classes, transform=True)

                    # Features
                    image_features, text_features = _get_normalized_CLIP_features(
                        img, text_inputs, self.model, self.device
                    )

                    # Logits
                    logit_scale = self.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.t()

                    guess = self._get_guess(logits_per_image, classes=classes)

                    self.logs[log_guess] += guess
                    self.logs[log_truth] += img_class

                    for i in range(len(img_class)):
                        if img_class[i] == guess[i]:
                            self.logs[log_correct].append(1)
                        else:
                            self.logs[log_correct].append(0)

                    if self.training:
                        # Tracking-Loss
                        text_inputs = self.set_text_inputs(
                            label=img_class, transform=False
                        )

                        # CLIP Features
                        image_features, text_features = _get_normalized_CLIP_features(
                            img, text_inputs, self.model, self.device
                        )

                        # CLIP Logits
                        logits_per_image = (
                            self.model.logit_scale.exp()
                            * image_features
                            @ text_features.t()
                        )
                        logits_per_text = (
                            self.model.logit_scale.exp()
                            * text_features
                            @ image_features.t()
                        )

                        # CLIP LOSS
                        loss = self.clip_loss(
                            logits_per_image, logits_per_text)

                        self.logs[log_loss] += loss

            if self.training:
                self.logs[log_loss] /= size_of_tested_set
            self.logs[log_acc] = sum(self.logs[log_correct]) / len(
                self.logs[log_correct]
            )

            # print('len_dataloader of ', log_acc, ': ', len(self.logs[log_correct]))
            print(log_acc, " is : " + str(self.logs[log_acc]))

    def _get_guess(self, logits_per_image, classes):
        probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
        self.logs["test_all_probs"] += [list(probs[x])
                                        for x in range(len(probs))]

        # Indize von der höchsten Warhscheinlkeit
        pred_batch = [self._argmax(probs[x]) for x in range(len(probs))]
        guess = [classes[pred_batch[x]] for x in range(len(probs))]

        return guess

    def weight_space_ensambling(self, alpha=0.5):
        """
        alpha=0: only finetuned model
        alpha=1: only zeroshot model
        """
        self.model.load_state_dict(wse(self.model, self.model_untuned, alpha))

    def _convert_models_to_fp32(self):
        """
        https://github.com/openai/CLIP/issues/57
        for mix precion training - Tutorial:
        https://developer.nvidia.com/blog/video-mixed-precision-techniques-tensor-cores-deep-learning/
        """
        for p in self.model.parameters():
            if p.data is not None:
                p.data = p.data.float()
            if hasattr(p.grad, "data"):
                p.grad.data = p.grad.data.float()

    def _log(self, logging_list):
        # log for wandb
        if self.wandb_log:
            wandb_log = {}
            for log_arg in logging_list:
                wandb_log[log_arg] = self.logs[log_arg]
            wandb.log(wandb_log)

    def _log_detailed_on_testset(self):
        self.test()
        self._log_detailed("test")

    def _log_detailed_on_valset(self):
        self.val()
        self._log_detailed("val")

    def _log_detailed(self, testset):
        # detailed log for wandb
        if testset == "test":
            imgs_l = "test_imgs"
            guess_l = "test_guess"
            truth_l = "test_truth"
            all_probs_l = "test_all_probs"
            guess_data_table = "guess_on_test_data"

            classes = self.testclasses

        elif testset == "val":
            imgs_l = "val_imgs"
            guess_l = "val_guess"
            truth_l = "val_truth"
            all_probs_l = "val_all_probs"
            guess_data_table = "guess_on_val_data"

            classes = self.valclasses
        else:
            print("function _log_detailed got wrong testset information")

        columns = ["image", "guess", "truth"]
        for a in classes:
            columns.append("score_" + a)

        pandas_df = pd.DataFrame(columns=columns)
        for i, g, t, a_p in tqdm(
            zip(
                self.logs[imgs_l],
                self.logs[guess_l],
                self.logs[truth_l],
                self.logs[all_probs_l],
            ),
            total=len(self.logs[guess_l]),
        ):

            if self.wandb_log:
                row = [wandb.Image(i), g, t]
            else:
                row = [None, g, t]
            for p in a_p:
                row.append(np.round(p, 4))

            pandas_df.loc[pandas_df.shape[0]] = row
        if self.wandb_log:
            wandb_df = wandb.Table(dataframe=pandas_df)
            wandb.log({guess_data_table: wandb_df})

    def _rs(self, image):
        # rescale Image
        return (image - image.min()) / (image.max() - image.min())

    def _argmax(self, iterable):
        return max(enumerate(iterable), key=lambda x: x[1])[0]

    def _init_logs(self):
        self._clear_train_logs()
        self._clear_val_logs()
        self._clear_test_logs('test')
        self._clear_test_logs('test_seen')
        self._clear_test_logs('test_unseen')

    def _clear_train_logs(self):
        self.logs["train_correct"] = []
        self.logs["train_guess"] = []
        self.logs["train_truth"] = []
        self.logs["train_imgs"] = []
        self.logs["Train Loss"] = 0
        self.logs["Train Accuracy"] = 0
        self.logs["Train Prompt Input for training"] = []
        self.logs["clip_loss"] = 0
        self.logs["simclr_loss"] = 0
        self.logs["robust_loss"] = 0

    def _clear_val_logs(self):
        self.logs["Val Loss"] = 0
        self.logs["Val Accuracy"] = 0
        self.logs["val_guess"] = []
        self.logs["val_correct"] = []
        self.logs["val_truth"] = []
        self.logs["val_imgs"] = []

    def _clear_test_logs(self, name_testsplit):
        self.logs[name_testsplit.capitalize() + " Loss"] = 0
        self.logs[name_testsplit.capitalize() + " Accuracy"] = 0
        self.logs[name_testsplit + "_guess"] = []
        self.logs[name_testsplit + "_correct"] = []
        self.logs[name_testsplit + "_imgs"] = []
        self.logs[name_testsplit + "_truth"] = []
        self.logs[name_testsplit + "_all_probs"] = []

    def set_text_inputs(self, label, transform=True):

        if transform:
            text_inputs = torch.cat(
                [clip.tokenize(self.transform_label(c).lower()) for c in label]
            )
        else:
            text_inputs = torch.cat([clip.tokenize(c) for c in label])
        return text_inputs

    def log_promt(self, label, transform=True):

        if transform:
            text_inputs = [self.transform_label(c).lower() for c in label]
        else:
            text_inputs = [c for c in label]
        return text_inputs
