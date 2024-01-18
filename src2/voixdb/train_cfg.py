from dataclasses import dataclass


@dataclass
class TrainerCfg:
    device: str
    epoch: int
    model_save_freq: int
    model_out_dir: str
