import wandb
import torch
import utils as uu
import utils_clip as uc
import utils_eu_moth as ue
import parameter_import as pi

from tqdm import tqdm
from trainer import Trainer


def main(args, config, model_architectur):

    device = "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"

    model, preprocess = uc.load_model(device, model_architectur)
    test_set, val_set, train_set = ue.load_data(args, preprocess)

    templates = ["a photo of a {}."]
    transform_label = templates[0].format

    if args.wandb_log:
        run_name = model_architectur.replace('/', '-')
        run = wandb.init(config=config, project=args.project, name=run_name)
        config = wandb.config
    else:
        config = uu.config_structure(**config)

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

    print("Zero-Shot Accuracy")
    trainer.val()
    trainer.test()
    trainer._log(["Test Loss", "Test Accuracy",
                  "Val Loss", "Val Accuracy"])

    if args.wandb_log:
        run.finish()

    return trainer.logs["Test Accuracy"]


if __name__ == "__main__":
    args, config, model_architectur = pi.load_parameter()
    main(args, config, model_architectur)
