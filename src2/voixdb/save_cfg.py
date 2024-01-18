import glob
import pathlib as pl
from dataclasses import dataclass
from typing import Optional


@dataclass
class SaveCfg:
    epoch: int
    out_dir: str
    eval_loss_4f: str

    out_dir_path: Optional[pl.Path] = None

    def __post_init__(self):
        self.out_dir_path = pl.Path(self.out_dir)

    def output_filename(self):
        if self.out_dir_path is None:
            raise ValueError()

        return self.out_dir_path / f"model_e{self.epoch}_ev{self.eval_loss_4f}.pth"

    def state_from_epoch(self):
        if self.out_dir_path is None:
            raise ValueError()

        for f in glob.glob(self.out_dir_path / f"model_e{self.epoch}_*"):
            return f
