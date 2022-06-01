import wandb
import torch
import parameter_import as pi
import utils_cub as ut
from trainer import Trainer
from sklearn.model_selection import train_test_split


def main(args, config, model_architectur, project='zero-shot-prediction-cub-batch-test', run_name_add=''):

    device = ("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    run_name = run_name_add + args.run_name

    run = wandb.init(
        config=config, project=project, name=run_name
    )
    config = wandb.config

    # load model and data
    model, preprocess = ut.load_clip(device, model_architectur)
    cub_train_val, cub_test = ut.load_data(config.root, preprocess)
    train_data_loader, val_data_loader, test_data_loader = ut.get_dataloader_from_dataset(
        cub_train_val, cub_test, config.validation_split, batch_size=20)
    classes = cub_train_val.classes

    # choose promt label
    templates = ["a photo of a {}."]
    transform_label = templates[0].format

    trainer = Trainer(
        model=model,
        classes=classes,
        config=config,
        device=device,
        wandb_log=args.wandb_log,
        transform_label=transform_label,
        training=False
    )

    trainer.set_data_loader(
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        test_data_loader=test_data_loader,
    )

    trainer.test()
    trainer._log(["Test Accuracy", "test_guess", "test_correct", "test_truth"])
    # trainer._log_test_detailed()

    # trainer.val()
    # trainer._log(["Val Accuracy"])

    run.finish()


if __name__ == "__main__":
    args, config, model_architectur = pi.load_parameter()
    main(args, config, model_architectur)
