import logging
import os
from functools import partial
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning.callbacks import ModelCheckpoint  # ← 新增

from fdiff.dataloaders.datamodules import Datamodule
from fdiff.models.score_models import ScoreModule
from fdiff.utils.callbacks import SamplingCallback
from fdiff.utils.extraction import dict_to_str, get_training_params
from fdiff.utils.wandb import maybe_initialize_wandb

# --- safe instantiate helpers ---
from pydoc import locate
from hydra.utils import instantiate as _hydra_instantiate
from typing import Any


def _build_datamodule(cfg: DictConfig) -> Any:
    dm_cfg = cfg.datamodule
    PASS_KEYS = (
        "data_dir", "batch_size", "window_length", "pred_len", "stride",
        "standardize", "fourier_transform",
        "num_workers", "pin_memory", "val_ratio",
        "jsonl_train", "jsonl_test",
    )

    # 1) Hydra 原生 _target_
    if isinstance(dm_cfg, DictConfig) and "_target_" in dm_cfg:
        return _hydra_instantiate(dm_cfg)

    # 2) 形如 {datamodule: "<类路径>", 其他参数...}
    if isinstance(dm_cfg, DictConfig) and "datamodule" in dm_cfg:
        inner = dm_cfg.get("datamodule")
        if isinstance(inner, DictConfig) and "_target_" in inner:
            return _hydra_instantiate(inner)
        if isinstance(inner, str):
            cls = locate(inner)
            if cls is None:
                raise RuntimeError(f"Cannot locate datamodule class: {inner}")
            params = {}
            for k in PASS_KEYS:
                if k in dm_cfg:
                    params[k] = dm_cfg.get(k)
                elif k in cfg:
                    params[k] = cfg.get(k)
            return cls(**params)

    # 3) 纯类路径字符串
    if isinstance(dm_cfg, str):
        cls = locate(dm_cfg)
        if cls is None:
            raise RuntimeError(f"Cannot locate datamodule class: {dm_cfg}")
        params = {}
        for k in PASS_KEYS:
            if k in cfg:
                params[k] = cfg.get(k)
        return cls(**params)

    # 4) 尝试旧工厂（向后兼容）
    try:
        from fdiff.dataloaders.datamodules import Datamodule as _Factory
        return _Factory(cfg)
    except Exception:
        pass

    # 5) 兜底
    return dm_cfg


class TrainingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        # Initialize torch
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        # Read out the config
        logging.info(
            f"Welcome in the training script! You are using the following config:\n{dict_to_str(cfg)}"
        )

        # Maybe initialize wandb
        run_id = maybe_initialize_wandb(cfg)

        # Instantiate all the components
        self.score_model: ScoreModule = instantiate(cfg.score_model)
        self.trainer: pl.Trainer = instantiate(cfg.trainer)
        self.datamodule: Datamodule = _build_datamodule(cfg)
        logging.info(f"[debug] datamodule instance type: {type(self.datamodule)}")

        # ---- 取 W&B 的真实 run id（在线时覆盖 offline-xxxx）----
        wb_id = None
        loggers = self.trainer.logger if isinstance(self.trainer.logger, (list, tuple)) else [self.trainer.logger]
        for lg in loggers:
            try:
                exp = getattr(lg, "experiment", None)  # WandbLogger.experiment -> wandb.Run
                if exp is not None and getattr(exp, "id", None):
                    wb_id = exp.id
                    break
            except Exception:
                pass
        if wb_id:
            run_id = wb_id

        # ---- 保存目录：支持 FDIFF_LOG_DIR 环境变量 ----
        log_root = Path(os.environ.get("FDIFF_LOG_DIR", str(Path.cwd() / "lightning_logs")))
        self.save_dir = log_root / run_id
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving the config into {self.save_dir}.")
        OmegaConf.save(config=cfg, f=self.save_dir / "train_config.yaml")

        # ---- 强制 ModelCheckpoint 把 ckpt 写到同一处 ----
        # Configure probe callback (optional)
        probe_cfg_dict = None
        if "probe" in cfg and cfg.probe is not None:
            probe_cfg_dict = OmegaConf.to_container(cfg.probe, resolve=True)
        if probe_cfg_dict is not None:
            enable = bool(probe_cfg_dict.get("enable", False))
            script_val = probe_cfg_dict.get("script")
            if enable and not script_val:
                default_script = Path(__file__).with_name("cov_compare_FourierDiffusion.py")
                if default_script.exists():
                    script_val = str(default_script)
                else:
                    raise FileNotFoundError("probe.enable=true 但未提供 probe.script，且默认脚本不存在")
            truth_jsonl = probe_cfg_dict.get("truth_jsonl")
            truth_npy = probe_cfg_dict.get("truth_npy")
            truth_csv = probe_cfg_dict.get("truth_csv")
            outdir = probe_cfg_dict.get("outdir")
            probe_dir = probe_cfg_dict.get("probe_dir")
            probe_eval_dir = probe_cfg_dict.get("probe_eval_dir")
            tag = probe_cfg_dict.get("tag")
            every = probe_cfg_dict.get("every")
            for cb in getattr(self.trainer, "callbacks", []):
                if isinstance(cb, SamplingCallback):
                    cb.configure_probe(
                        enable=enable,
                        script=script_val,
                        truth_jsonl=truth_jsonl,
                        truth_npy=truth_npy,
                        truth_csv=truth_csv,
                        out_root=outdir,
                        probe_dir=probe_dir,
                        probe_eval_dir=probe_eval_dir,
                        tag=tag,
                        every=every,
                    )

        for cb in getattr(self.trainer, "callbacks", []):
            if isinstance(cb, ModelCheckpoint):
                cb.dirpath = str(self.save_dir / "checkpoints")
                os.makedirs(cb.dirpath, exist_ok=True)

        # Set-up dataset
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")

        # Finish instantiation of the model if necessary
        if isinstance(self.score_model, partial):
            training_params = get_training_params(self.datamodule, self.trainer)
            self.score_model = self.score_model(**training_params)

        # Possibly setup the datamodule in the sampling callback
        for callback in self.trainer.callbacks:  # type: ignore
            if isinstance(callback, SamplingCallback):
                callback.setup_datamodule(datamodule=self.datamodule)

    def train(self) -> None:
        assert not (
            self.score_model.scale_noise and not self.datamodule.fourier_transform
        ), "You cannot use noise scaling without the Fourier transform."
        self.trainer.fit(model=self.score_model, datamodule=self.datamodule)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    runner = TrainingRunner(cfg)
    runner.train()


if __name__ == "__main__":
    main()
