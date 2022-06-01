import wandb
import torch
import utils as uu
import utils_cub as ut
import parameter_import as pi
from tqdm import tqdm

from datasetEuMoths.eu_moths import eu_moths
import utils_eu_moth as emu
from trainer import Trainer
from sklearn.model_selection import train_test_split


def main(args, config, model_architectur, project="split_finetuning_on_eu_moths"):

    device = "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    run_name = (
        args.label + "_" + args.testsetting + "_" + args.run_name
    )

    model, preprocess = ut.load_clip(device, model_architectur)

    # load data in one
    if args.label == "english":
        all_moths = eu_moths(
            args.root,
            transform=preprocess,
            datasplit=None,
            label=args.label,
            path_engl=args.label,
        )
        all_moths.data = all_moths.data.reset_index()
        all_moths.data.drop("index", axis=1, inplace=True)
    else:
        all_moths = eu_moths(
            args.root, transform=preprocess, datasplit=None, label=args.label
        )

    # split data in train-, val- and testset (zero-shot-setting)
    train_indices = all_moths.data[all_moths.data.label < 100].index.tolist()
    test_seen_indices = all_moths.data[
        (all_moths.data.label >= 100) & (all_moths.data.label < 150)
    ].index.tolist()
    test_unseen_indices = all_moths.data[all_moths.data.label >= 150].index.tolist(
    )
    train_dataset, test_seen_dataset, test_unseen_dataset = emu.split_dataset(
        all_moths, train_indices, test_seen_indices, test_unseen_indices
    )

    if args.testsetting == "zsl":
        classes = None
        trainclasses = all_moths.classes[:100]
        test_seen_classes = all_moths.classes[100:150]
        test_unseen_classes = all_moths.classes[150:200]

    else:
        classes = all_moths.classes
        trainclasses = all_moths.classes
        test_seen_classes = all_moths.classes
        test_unseen_classes = all_moths.classes

    templates = ["a photo of a {}."]
    transform_label = templates[0].format

    run = wandb.init(config=config, project=project, name=run_name)
    config = wandb.config

    trainer = Trainer(
        model=model,
        classes=classes,
        test_seen_classes=test_seen_classes,
        test_unseen_classes=test_unseen_classes,
        trainclasses=trainclasses,
        config=config,
        device=device,
        wandb_log=args.wandb_log,
        transform_label=transform_label,
        training=True,
    )
    train_data_loader, test_seen_data_loader, test_unseen_data_loader = emu.get_dataloader(
        train_dataset, test_seen_dataset, test_unseen_dataset, batch_size=config.batch_size
    )

    trainer.set_data_loader(
        train_data_loader=train_data_loader,
        test_seen_data_loader=test_seen_data_loader,
        test_unseen_data_loader=test_unseen_data_loader,
    )

    if config.scheduler == "yes":
        trainer.set_scheduler()

    print("Zero-Shot Accuracy")
    trainer.test('seen')
    trainer._log(["Test_seen Loss", "Test_seen Accuracy"])

    # print("Start Training")
    for epoch in tqdm(range(1, trainer.config.epochs + 1)):
        trainer.train()
        trainer._log(["Train Loss", "Train Accuracy",
                     "Val Loss", "Val Accuracy"])

    trainer.test('seen')
    trainer.test('unseen')
    trainer._log([
        "Test_seen Loss",
        "Test_seen Accuracy",
        "Test_unseen Loss",
        "Test_unseen Accuracy",
    ])
    # trainer._log_test_detailed()
    wandb.log(vars(args))

    run.finish()
    return trainer.logs["Test Accuracy"]


if __name__ == "__main__":
    args, config, model_architectur = pi.load_parameter()
    main(args, config, model_architectur)
