import argparse
import os
from EDGE import EDGE
from omegaconf import OmegaConf
import wandb
from pathlib import Path



os.environ["WANDB_API_KEY"] = "8703d7348effc00c329d216eb3eed220e0e2912d"
os.environ["WANDB_MODE"] = "online"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# cfg = OmegaConf.merge(OmegaConf.load("configs/data.yaml"),  OmegaConf.load("configs/train.yaml"))
# cfg = OmegaConf.create(vars(cfg))
def train(cfg):
    model = EDGE(cfg, cfg.TRAIN.PRETRAINED)
    model.train_loop(cfg)


if __name__ == "__main__":
    # cfg = parse_train_cfg()
    parser = argparse.ArgumentParser()
    # parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    # parser.add_argument("--epochs", type=int, default=2000)
    # parser.add_argument("--feature_type", type=str, default="baseline")  # jukebox

    parser.add_argument("--cfg", type=str,
            required=False,
            default="./configs/aist/fea266.yaml",
            help="config file",
        )
    parser.add_argument("--assets", type=str,
            required=False,
            default="./configs/45assets.yaml",
            help="config file for asset paths",
        )
    args = parser.parse_args()
    args = OmegaConf.create(vars(args))
    cfg_exp = OmegaConf.load(args.cfg)
    cfg_assets = OmegaConf.load(args.assets)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)
    cfg = OmegaConf.merge(cfg, args)

    
    train(cfg)
