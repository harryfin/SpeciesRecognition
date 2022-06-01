import wandb
import torch
import utils_cub as ut
import utils as uu
import parameter_import as pi
from tqdm import tqdm
from trainer import Trainer
from datasetCUB.Cub_class.class_cub import Cub2011


def main(args, config, model_architectur, project="complete-clip-finetuning-on-cub"):

    device = "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    run_name = args.split + "_" + args.run_name

    model, preprocess = ut.load_clip(device, model_architectur)

    preprocess_train = preprocess

    cub_root = uu.get_root_CUB(args.root)

    cub_train_val_preprocess_train = Cub2011(
        root=cub_root, train=True, transform_image=preprocess_train
    )
    cub_train_val_preprocess_val = Cub2011(
        root=cub_root, train=True, transform_image=preprocess
    )
    cub_test = Cub2011(root=cub_root, train=False, transform_image=preprocess)

    classes = cub_test.classes

    templates = ["a photo of a {}."]
    transform_label = templates[0].format

    if args.wandb_log:
        run = wandb.init(config=config, project=project, name=run_name)
        config = wandb.config
    else:
        config = uu.config_structure(**config)

    trainer = Trainer(
        model=model,
        classes=classes,
        config=config,
        device=device,
        wandb_log=args.wandb_log,
        validation=True,
        transform_label=transform_label,
        training=True,
    )

    (
        train_data_loader,
        val_data_loader,
        test_data_loader,
    ) = ut.get_dataloader_from_dataset_with_different_preprocesses(
        cub_train_val_preprocess_train,
        cub_train_val_preprocess_val,
        cub_test,
        config.validation_split,
        config.batch_size,
    )

    trainer.set_data_loader(
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        test_data_loader=test_data_loader,
    )

    if config.scheduler == "yes":
        trainer.set_scheduler()

    print("Zero-Shot Accuracy")
    trainer.test()
    trainer._log(["Test Loss", "Test Accuracy"])

    print("Start Training")
    for epoch in tqdm(range(1, trainer.config.epochs + 1)):
        trainer.train()
        trainer.val()
        trainer._log(["Val Loss", "Val Accuracy",
                     "Test Loss", "Test Accuracy"])

    print("Accuracy after Training")
    trainer.test()
    trainer._log(["Test Loss", "Test Accuracy",
                 "Train Prompt Input for training"])

    if args.wandb_log:
        # trainer._log_test_detailed()
        run.finish()

    return trainer.logs["Test Accuracy"]


if __name__ == "__main__":
    args, config, model_architectur = pi.load_parameter()
    main(args, config, model_architectur)
