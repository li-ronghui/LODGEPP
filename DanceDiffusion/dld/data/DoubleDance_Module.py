import numpy as np
import torch

# from mld.data.humanml.scripts.motion_process import (process_file,
#                                                      recover_from_ric)

# from .BaseData_Module import BASEDataModule
from .FineDance_dataset import FineDance_Smpl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from dld.data.utils.smplfk import SMPLX_Skeleton
from .utils.preprocess import ax_from_6v
from .FineDance_dataset import FineDance_Smpl, DoubleDance
from .DoubleDance_dataset import DoubleDance_split

# dataset_map = {
#     "finedance": FineDance_Smpl,
#     "finedance_263cut": FineDance_Smpl,
#     "doubledance_263": DoubleDance,
# }

class DoubleDanceModule(pl.LightningDataModule):
    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 name,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.name = name
        self.kwargs = kwargs
        self.is_mm = False
        
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = DoubleDance_split(args=self.cfg, istrain=True, dataname=self.name)
            self.valset = DoubleDance_split(args=self.cfg, istrain=False, dataname=self.name)
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = DoubleDance_split(args=self.cfg, istrain=False, dataname=self.name)
        
            
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        # batch_size=self.cfg.EVAL.BATCH_SIZE, num_workers=self.cfg.EVAL.NUM_WORKERS
        return DataLoader(self.valset, batch_size=self.cfg.EVAL.BATCH_SIZE, num_workers=self.cfg.EVAL.NUM_WORKERS, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.cfg.EVAL.BATCH_SIZE, num_workers=self.cfg.EVAL.NUM_WORKERS, shuffle=False)

    
if __name__ == "__main__":
    trainset = FineDance_Smpl(args={'a':1}, istrain=True)

