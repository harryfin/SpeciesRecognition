import torch
import wandb
import utils as uu
import utils_cub as ut
import parameter_import as pi
import datasetCUB.transformations.image_transformation as transform

from trainer import Trainer
from tqdm import tqdm
from datasetCUB.Cub_class.class_cub_mp_splits import class_cub_mp_split


def main(args, config, model_architectur, project="split_clip_finetuning_on_cub"):

    device = "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    run_name = args.add_to_run_name + args.split + "_" + args.run_name

    # init wandb
    if args.wandb_log:
        run = wandb.init(config=config, project=project, name=run_name)
        wandb.config.update(
            {
                "simclr": args.simclr,
                "clip_scale": args.clip_scale,
                "robust_mse": args.robust_MSE,
                "robust_wse_end": args.robust_WSE_end,
                "robust_wse_always": args.robust_WSE_always,
            })
        if args.robust_WSE_always or args.robust_WSE_end:
            wandb.config.update({"alpha": args.alpha})
        if args.robust_MSE:
            wandb.config.update({"robust_scale": args.robust_scale})
        if args.simclr:
            wandb.config.update({"simclr_scale": args.simclr_scale})
        config = wandb.config
    else:
        config = uu.config_structure(**config)

    # load model und preprocesses
    model, preprocess = ut.load_clip(
        device, model_architectur, config.seed, config.cuda_det
    )

    if args.simclr_preprocess == "slip":
        preprocess_simclr = (
            transform.preprocess_with_data_augmentation_for_simclr_training_slip(
                model)
        )
    elif args.simclr_preprocess == "own":
        preprocess_simclr = (
            transform.preprocess_with_data_augmentation_for_simclr_training(
                model)
        )
    else:
        print("no preprocess for simclr choosen")

    if args.clip_data_augmentation:
        preprocess_data_augmentation = (
            transform.preprocess_with_data_augmentation_for_clip_training(
                model)
        )

    # load data
    if args.simclr:
        dataset_class = class_cub_mp_split(
            args.root,
            args.root_cub,
            preprocess_clip=preprocess,
            preprocess_simclr=preprocess_simclr,
            split=args.split,
            simclr_augmentation=True,
        )
        print("Process with SimCLR")
    elif args.clip_data_augmentation:
        dataset_class = class_cub_mp_split(
            args.root,
            args.root_cub,
            preprocess_clip=preprocess,
            preprocess_training=preprocess_data_augmentation,
            split=args.split,
            data_augmentation=True,
        )
        print("Process with data augmentation")
    else:
        dataset_class = class_cub_mp_split(
            args.root, args.root_cub, preprocess_clip=preprocess, split=args.split)
        print("CLIP Preprocess")
    all_classes = dataset_class.cub.classes

    if args.split == "SS":
        trainclasses = dataset_class.train_class_SS
        test_seen_classes = None
        test_unseen_classes = dataset_class.test_unseen_class_SS

    else:
        trainclasses = dataset_class.train_class_PS
        test_seen_classes = all_classes
        test_unseen_classes = all_classes

    # set template
    templates = ["a photo of a {}."]
    transform_label = templates[0].format

    # init Trainer and do Trainer Settings
    trainer = Trainer(
        model=model,
        test_seen_classes=test_seen_classes,
        test_unseen_classes=test_unseen_classes,
        trainclasses=trainclasses,
        config=config,
        device=device,
        wandb_log=args.wandb_log,
        transform_label=transform_label,
        training=True,
        robust_loss_MSE=args.robust_MSE,
        wse=(args.robust_WSE_always or args.robust_WSE_end),
        simclr=args.simclr,
        training_data_augmentation=args.clip_data_augmentation,
        clip_scale=args.clip_scale,
        simclr_scale=args.simclr_scale,
        robust_scale=args.robust_scale,
    )
    (
        trainval_data_loader,
        test_unseen_data_loader,
        test_seen_data_loader,
    ) = dataset_class.get_dataloader(config.batch_size)

    trainer.set_data_loader(
        train_data_loader=trainval_data_loader,
        test_seen_data_loader=test_seen_data_loader,
        test_unseen_data_loader=test_unseen_data_loader,
    )

    # set scale
    trainer.clip_scale = args.clip_scale
    trainer.simclr_scale = args.simclr_scale
    trainer.robust_scale = args.robust_scale

    if config.scheduler == "yes":
        trainer.set_scheduler()

    # Zero Shot results
    if args.split == "PS":
        print("Zero-Shot Accuracy on test_seen_data")
        trainer.test('seen')
        trainer._log(["Test_seen Loss", "Test_seen Accuracy"])

    print("Zero-Shot Accuracy on test_unseen_data")
    trainer.test('unseen')
    trainer._log(["Test_unseen Loss", "Test_unseen Accuracy"])

    print("Start Training")
    for epoch in tqdm(range(1, trainer.config.epochs + 1)):
        trainer.train()
        if args.robust_WSE_always:
            trainer.weight_space_ensambling(alpha=args.alpha)

    if args.robust_WSE_end:
        trainer.weight_space_ensambling(alpha=args.alpha)
        print("\n\nafter weight_space_ensambling\n\n")

    if args.split == "PS":
        trainer.test('seen')
    trainer.test('unseen')

    if args.split == "PS":
        logginglist = [
            "Test_seen Loss",
            "Test_seen Accuracy",
            "Test_unseen Loss",
            "Test_unseen Accuracy",
            "Train Prompt Input for training",
        ]
    else:
        logginglist = [
            "Test_unseen Loss",
            "Test_unseen Accuracy",
            "Train Prompt Input for training",
        ]
    trainer._log(logginglist)
    # trainer._log_test_detailed()

    if args.wandb_log:
        run.finish()

    return trainer.logs["Test_seen Accuracy"], trainer.logs["Test_unseen Accuracy"]


if __name__ == "__main__":
    args, config, model_architectur = pi.load_parameter()
    main(args, config, model_architectur)
