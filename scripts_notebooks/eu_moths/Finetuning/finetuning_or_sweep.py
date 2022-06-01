import wandb
import torch
import utils_clip as uc
import utils_eu_moth as ue
import parameter_import as pi

from tqdm import tqdm
from trainer import Trainer


def train():
    templates = ["a photo of a {}."]
    transform_label = templates[0].format

    test_set, val_set, train_set = data
    model, _ = uc.load_model(device, model_architectur)

    with wandb.init(config=config_parameter):
        config = wandb.config

        trainer = Trainer(
            model=model,
            classes=train_set.classes,
            config=config,
            device=device,
            wandb_log=args.wandb_log,
            transform_label=transform_label,
        )

        train_data_loader = torch.utils.data.DataLoader(
            train_set, batch_size=config.batch_size, shuffle=True
        )
        val_data_loader = torch.utils.data.DataLoader(
            val_set, batch_size=config.batch_size, shuffle=True
        )
        test_data_loader = torch.utils.data.DataLoader(
            test_set, batch_size=config.batch_size, shuffle=True
        )

        trainer.set_data_loader(
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            test_data_loader=test_data_loader,
        )

        if config.scheduler == "yes":
            trainer.set_scheduler()

        print("Zero-Shot Accuracy")
        trainer.val()
        trainer._log(["Val Loss", "Val Accuracy"])
        print("Start Training")
        for epoch in tqdm(range(1, trainer.config.epochs + 1)):
            trainer.train()
            trainer.val()
            trainer._log(["Train Loss", "Train Accuracy",
                         "Val Loss", "Val Accuracy"])

        trainer.test()
        trainer._log(["Test Loss", "Test Accuracy"])
        return trainer.logs["Test Accuracy"]


def main():
    global args, config_parameter, model_architectur, device, data
    args, config_parameter, model_architectur = pi.load_parameter()
    device = "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    _, preprocess = uc.load_model(device, model_architectur)
    data = ue.load_data(args, preprocess)

    if args.sweep:
        sweep_id = wandb.sweep(config_parameter, project=args.project_name)
        wandb.agent(sweep_id, train, count=args.sweep_runs)
    else:
        return train()


if __name__ == "__main__":
    main()
