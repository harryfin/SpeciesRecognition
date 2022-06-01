# Argparser
import argparse
from datetime import datetime
import yaml
import utils as ut
import sys
import os.path
import clip


def args_import():
    parser = argparse.ArgumentParser(description="Clip-Finetuning")
    parser.add_argument(
        "root", help="path to root of repository", type=str
    )
    parser.add_argument(
        "yaml_config", help="path to config.yaml file for config parameters", type=str
    )
    # wandb settings
    parser.add_argument(
        "--sweep", type=str, default="no", help="Hyperparameter Sweep: 'yes' or 'no'"
    )
    parser.add_argument(
        "--add_to_run_name",
        type=str,
        default="",
        help="add to something special to run name",
    )
    parser.add_argument(
        "--wandb_log", type=str, default="yes", help="log in wandb: yes or no"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu: 0 default")
    parser.add_argument("--runs", type=int, default=1,
                        help="how many runs? default: 1")
    parser.add_argument(
        "--sweep_runs",
        type=int,
        default=10,
        help="how many runs on a sweep: 10 default",
    )
    parser.add_argument(
        "--project_name", type=str, default="test", help="wandb projectname"
    )

    # for split eu_moth dataset
    parser.add_argument(
        "--split",
        type=str,
        default="PS",
        help="Splitvariante: SS or PS (standard split or proposed split) ",
    )
    parser.add_argument(
        "--testsetting", type=str, default="zsl", help="Test-Setting: zsl or gzsl"
    )
    # CLIP augmentation
    parser.add_argument(
        "--clip_data_augmentation",
        type=str,
        default=None,
        help="data augmentation for clip",
    )
    parser.add_argument("--simclr", type=str, default=None, help="simclr")
    parser.add_argument("--simclr_preprocess", type=str,
                        default=None, help="slip")

    # CLIP robust learning
    parser.add_argument(
        "--robust_WSE_always",
        type=str,
        default=None,
        help="weight space ensamling each iteration",
    )
    parser.add_argument(
        "--robust_WSE_end",
        type=str,
        default=None,
        help="weight space ensamling at the end",
    )
    parser.add_argument("--robust_MSE", type=str, default=None,
                        help="robust finetuning with MSE loss function")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha")

    # scaling Losses
    parser.add_argument("--clip_scale", type=float,
                        default=1.0, help="clip_scale")
    parser.add_argument("--simclr_scale", type=float,
                        default=1.0, help="simclr_scale")
    parser.add_argument("--robust_scale", type=float,
                        default=1.0, help="robust_scale")

    # CUB split dataset
    parser.add_argument("--train", type=str, default=None, help="trainsplit: ")
    parser.add_argument("--val", type=str, default=None, help="valsplit")
    parser.add_argument("--test", type=str, default=None, help="testsplit")

    # eu_moths dataset
    parser.add_argument(
        "--label", type=str, default="english", help="label: biological or english"
    )
    args = parser.parse_args()
    return args


def load_parameter():

    args = args_import()
    config_path = os.path.join(args.root, args.yaml_config)
    with open(config_path, "r") as stream:
        yaml_config = yaml.safe_load(stream)

    # create boolean parameters
    list_of_false = [None, "false", "False", "no", "No"]
    args.wandb_log = True if args.wandb_log not in list_of_false else False
    args.simclr = True if args.simclr not in list_of_false else False
    args.robust_MSE = True if args.robust_MSE not in list_of_false else False
    args.robust_WSE_always = (
        True if args.robust_WSE_always not in list_of_false else False
    )
    args.robust_WSE_end = True if args.robust_WSE_end not in list_of_false else False
    args.clip_data_augmentation = (
        True if args.clip_data_augmentation not in list_of_false else False
    )
    args.sweep = True if args.sweep not in list_of_false else False

    # checking parameters for correctness
    if args.sweep:
        args.wandb_log = True
        try:
            model_architecture = yaml_config["parameters"]["model_architecture"]["value"]
        except KeyError:
            print('wrong config file')
    else:
        model_architecture = yaml_config["model_architecture"]

    model_list = clip.available_models()

    if model_architecture not in model_list:
        print("model_architecture: ", model_architecture)
        sys.exit(
            "Error: available model architecture: ", model_list)

    if args.label not in ["english", "biological"]:
        print("Label: ", args.label)
        sys.exit("args: english" or "biological")

    # add dataset paths to args
    args.root_cub = ut.get_root_CUB(args.root)
    args.root_eu_moths = ut.get_root_eu_moths(args.root)
    args.root_mmc = ut.get_root_mmc(args.root)

    # edit run name
    now = datetime.now()
    date_time = now.strftime("%d.%m.%Y-%H:%M:%S")
    args.run_name = model_architecture.replace(
        "/", "-").lower() + "_" + date_time

    return args, yaml_config, model_architecture
