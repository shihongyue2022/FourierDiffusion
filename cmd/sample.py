import logging
from pathlib import Path
from typing import Any

import hydra
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pydoc import locate
from hydra.utils import instantiate as _hydra_instantiate

from fdiff.dataloaders.datamodules import Datamodule
from fdiff.models.score_models import ScoreModule  # noqa: F401 (kept for Hydra instantiate)
from fdiff.sampling.metrics import MetricCollection
from fdiff.sampling.sampler import DiffusionSampler
from fdiff.utils.extraction import dict_to_str, get_best_checkpoint, get_model_type
from fdiff.utils.fourier import idft


# ------- helpers: 与 train.py 对齐的 datamodule 构造 -------
def _build_datamodule_from_train_cfg(cfg: DictConfig) -> Any:
    """
    从保存下来的 train_config.yaml 中构造 datamodule 实例。
    兼容三种写法：
      1) datamodule 是 _target_ dict
      2) datamodule 是 {datamodule: "包.类路径", 其它参数...}
      3) datamodule 是 "包.类路径"
    参数优先级：顶层 cfg 覆盖 > datamodule 节点默认。
    """
    dm_cfg = cfg.get("datamodule")
    PASS_KEYS = (
        "data_dir", "batch_size", "window_length", "pred_len", "stride",
        "standardize", "fourier_transform",
        "num_workers", "pin_memory", "val_ratio",
        "jsonl_train", "jsonl_test",
    )

    # 1) 原生 _target_
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
            # 关键：先用顶层 cfg 覆盖，再用 dm_cfg 默认
            for k in PASS_KEYS:
                if k in cfg:
                    params[k] = cfg.get(k)
                elif k in dm_cfg:
                    params[k] = dm_cfg.get(k)
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

    # 4) 兜底：也许传进来的已经是实例/字典
    return dm_cfg


class SamplingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        # Initialize torch
        self.random_seed: int = cfg.random_seed
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        # Read out the config
        logging.info(
            f"Welcome in the sampling script! You are using the following config:\n{dict_to_str(cfg)}"
        )

        # Get model path and id
        self.model_path = Path(cfg.model_path)
        self.model_id = cfg.model_id

        # Save sampling config to model directory
        self.save_dir = self.model_path / self.model_id
        self.save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=cfg, f=self.save_dir / "sample_config.yaml")

        # Read training config from model directory
        train_cfg = OmegaConf.load(self.save_dir / "train_config.yaml")

        # --------- 实例化 datamodule（更健壮） ----------
        self.datamodule: Datamodule = _build_datamodule_from_train_cfg(train_cfg)  # type: ignore
        # 准备数据
        if hasattr(self.datamodule, "prepare_data"):
            self.datamodule.prepare_data()
        if hasattr(self.datamodule, "setup"):
            try:
                self.datamodule.setup("test")
            except Exception:
                # 某些实现只接受 "fit"
                self.datamodule.setup("fit")  # type: ignore

        # fourier_transform：优先实例属性，其次回退到训练时顶层字段
        self.fourier_transform: bool = bool(
            getattr(self.datamodule, "fourier_transform",
                    bool(train_cfg.get("fourier_transform", False)))
        )

        # Get number of steps and samples
        self.num_samples: int = cfg.num_samples
        self.num_diffusion_steps: int = cfg.num_diffusion_steps

        # Load score model from checkpoint
        best_checkpoint_path = get_best_checkpoint(self.save_dir / "checkpoints")
        model_type = get_model_type(train_cfg)
        self.score_model = model_type.load_from_checkpoint(
            checkpoint_path=best_checkpoint_path
        )
        if torch.cuda.is_available():
            self.score_model.to(device=torch.device("cuda"))

        # Instantiate sampler
        sampler_partial = instantiate(cfg.sampler)
        self.sampler: DiffusionSampler = sampler_partial(score_model=self.score_model)

        # ------- 组装 original_samples 给 MetricCollection --------
        original_samples = None
        # 1) 直接用 datamodule.X_train（若存在且非空）
        if hasattr(self.datamodule, "X_train"):
            xs = getattr(self.datamodule, "X_train")
            if isinstance(xs, torch.Tensor) and xs.numel() > 0:
                original_samples = xs
        # 2) 从 train_dataloader 里抓一批
        if original_samples is None and hasattr(self.datamodule, "train_dataloader"):
            try:
                batch = next(iter(self.datamodule.train_dataloader()))  # type: ignore
                if isinstance(batch, dict) and "X" in batch:
                    original_samples = batch["X"]
                else:
                    # 兼容直接返回张量的 DataLoader
                    original_samples = batch
            except Exception:
                pass

        if original_samples is None:
            raise RuntimeError(
                "Unable to obtain original_samples for metrics. "
                "Please ensure your datamodule provides either `X_train` "
                "or a working `train_dataloader()`."
            )

        # Instantiate metrics
        metrics_partial = instantiate(cfg.metrics)
        self.metrics: MetricCollection = metrics_partial(original_samples=original_samples)

    def _inverse_standardize(self, X: torch.Tensor) -> torch.Tensor:
        """
        逆标准化：优先 datamodule.feature_mean_and_std，
        否则尝试 datamodule._mean/_std（numpy 或 tensor）
        """
        # datamodule 是否标准化
        dm_std = bool(getattr(self.datamodule, "standardize", False))
        if not dm_std:
            return X

        # 方案1：feature_mean_and_std
        if hasattr(self.datamodule, "feature_mean_and_std"):
            try:
                mean, std = self.datamodule.feature_mean_and_std  # type: ignore
                if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
                    return X * std + mean
            except Exception:
                pass

        # 方案2：_mean/_std（常见于我们自定义的 GluonTSJsonDatamodule）
        mean = getattr(self.datamodule, "_mean", None)
        std = getattr(self.datamodule, "_std", None)
        if mean is not None and std is not None:
            # 允许 numpy / list / tensor
            mean_t = torch.as_tensor(mean, dtype=X.dtype, device=X.device).view(1, 1, -1)
            std_t = torch.as_tensor(std, dtype=X.dtype, device=X.device).view(1, 1, -1)
            return X * std_t + mean_t

        # 无法还原就原样返回
        return X

    def sample(self) -> None:
        # Sample from score model
        X = self.sampler.sample(
            num_samples=self.num_samples, num_diffusion_steps=self.num_diffusion_steps
        )

        # 逆标准化（若训练时做过标准化）
        X = self._inverse_standardize(X)

        # 若在频域采样，则变回时域
        if self.fourier_transform:
            X = idft(X)

        # Compute metrics
        results = self.metrics(X)
        logging.info(f"Metrics:\n{dict_to_str(results)}")

        # Save everything
        logging.info(f"Saving samples and metrics to {self.save_dir}.")
        yaml.dump(
            data=results,
            stream=open(self.save_dir / "results.yaml", "w"),
        )
        torch.save(X, self.save_dir / "samples.pt")


@hydra.main(version_base=None, config_path="conf", config_name="sample")
def main(cfg: DictConfig) -> None:
    runner = SamplingRunner(cfg)
    runner.sample()


if __name__ == "__main__":
    main()
