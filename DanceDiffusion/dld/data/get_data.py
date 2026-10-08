from os.path import join as pjoin

import numpy as np
# from .humanml.utils.word_vectorizer import WordVectorizer
# from .HumanML3D import HumanML3DDataModule
# from .Kit import KitDataModule
# from .Humanact12 import Humanact12DataModule
# from .Uestc import UestcDataModule
# from .utils import *
from .FineDance_Module import FineDanceDataModule
from .DoubleDance_Module import DoubleDanceModule

# map config name to module&path
dataset_module_map = {
    "finedance": FineDanceDataModule,
    "finedance_263cut": FineDanceDataModule,
    "finedance_266cut": FineDanceDataModule,
    "finedance_338": FineDanceDataModule,
    "finedance_139cut": FineDanceDataModule,
    "doubledance_263": DoubleDanceModule,
    "doubledance_266": DoubleDanceModule,
    "aistpp": FineDanceDataModule,
    "aistpp_151": FineDanceDataModule,
    "aistpp_266": FineDanceDataModule,
    "aistpp_290": FineDanceDataModule,
    "aistpp_long263": FineDanceDataModule,
    "aistpp_60fps": FineDanceDataModule,
    "double_dance": DoubleDanceModule,
}
# motion_subdir = {"FineDance": "new_joint_vecs"}


def get_datasets(cfg, logger=None, phase="train"):
    # get dataset names form cfg
    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name.lower() in ["finedance", "finedance_263cut", "finedance_266cut", "finedance_139cut", "doubledance_263", "finedance_266cut", "finedance_338", "doubledance_266", "aistpp", "aistpp_151", "aistpp_266", "aistpp_290", "aistpp_long263", "aistpp_60fps", "double_dance"]:
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                name=dataset_name,
            )
            datasets.append(dataset)
      
    return datasets
