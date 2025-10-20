import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from fdiff.dataloaders.datamodules import Datamodule
from fdiff.models.score_models import ScoreModule
from fdiff.sampling.metrics import Metric, MetricCollection
from fdiff.sampling.sampler import DiffusionSampler

from typing import Optional

from .fourier import idft


class SamplingCallback(pl.Callback):
    def __init__(
        self,
        every_n_epochs: int,
        sample_batch_size: int,
        num_samples: int,
        num_diffusion_steps: int,
        metrics: list[Metric],
    ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.sample_batch_size = sample_batch_size
        self.num_samples = num_samples
        self.num_diffusion_steps = num_diffusion_steps
        self.metrics = metrics
        self.datamodule_initialized = False

        self.probe_enabled: bool = False
        self.probe_script: Optional[Path] = None
        self.probe_out_root: Optional[Path] = None
        self.probe_truth_jsonl: Optional[Path] = None
        self.probe_truth_npy: Optional[Path] = None
        self.probe_truth_csv: Optional[Path] = None
        self.probe_dir: Optional[Path] = None
        self.probe_eval_dir: Optional[Path] = None
        self.probe_tag_base: Optional[str] = None
        self.probe_every = self.every_n_epochs
        self._run_dir: Optional[Path] = None
        self._last_sample_elapsed: float = 0.0

        env_enable = os.getenv("FDIFF_PROBE_ENABLE", "").lower() in {"1", "true", "yes"}
        if env_enable:
            env_every = os.getenv("FDIFF_PROBE_EVERY")
            try:
                self.configure_probe(
                    enable=True,
                    script=os.getenv("FDIFF_PROBE_SCRIPT"),
                    truth_jsonl=os.getenv("FDIFF_PROBE_TRUTH_JSONL"),
                    truth_npy=os.getenv("FDIFF_PROBE_TRUTH_NPY"),
                    truth_csv=os.getenv("FDIFF_PROBE_TRUTH_CSV"),
                    out_root=os.getenv("FDIFF_PROBE_OUTDIR"),
                    probe_dir=os.getenv("FDIFF_PROBE_DIR"),
                    probe_eval_dir=os.getenv("FDIFF_PROBE_EVAL_DIR"),
                    tag=os.getenv("FDIFF_PROBE_TAG"),
                    every=int(env_every) if env_every else None,
                )
            except Exception as exc:
                print(f"[probe] 环境变量配置探针失败: {exc}")
                self.probe_enabled = False
                self.probe_script = None

    def configure_probe(
        self,
        *,
        enable: bool,
        script: Optional[str] = None,
        truth_jsonl: Optional[str] = None,
        truth_npy: Optional[str] = None,
        truth_csv: Optional[str] = None,
        out_root: Optional[str] = None,
        probe_dir: Optional[str] = None,
        probe_eval_dir: Optional[str] = None,
        tag: Optional[str] = None,
        every: Optional[int] = None,
    ) -> None:
        """Override probe configuration programmatically."""
        if not enable:
            self.probe_enabled = False
            self.probe_script = None
            self.probe_out_root = None
            self.probe_truth_jsonl = None
            self.probe_truth_npy = None
            self.probe_truth_csv = None
            self.probe_dir = None
            self.probe_eval_dir = None
            self.probe_tag_base = None
            self.probe_every = self.every_n_epochs
            return

        script_path = Path(script).expanduser().resolve() if script else None
        if script_path is None:
            raise ValueError("probe script must be provided when enable=True")
        if not script_path.exists():
            raise FileNotFoundError(f"probe script not found: {script_path}")

        def _opt_path(value: Optional[str]) -> Optional[Path]:
            if value:
                return Path(value).expanduser().resolve()
            return None

        truth_jsonl_path = _opt_path(truth_jsonl)
        if truth_jsonl_path and not truth_jsonl_path.exists():
            raise FileNotFoundError(f"truth_jsonl not found: {truth_jsonl_path}")

        truth_npy_path = _opt_path(truth_npy)
        if truth_npy_path and not truth_npy_path.exists():
            raise FileNotFoundError(f"truth_npy not found: {truth_npy_path}")

        truth_csv_path = _opt_path(truth_csv)
        if truth_csv_path and not truth_csv_path.exists():
            raise FileNotFoundError(f"truth_csv not found: {truth_csv_path}")

        out_root_path = _opt_path(out_root)
        probe_dir_path = _opt_path(probe_dir)
        probe_eval_dir_path = _opt_path(probe_eval_dir)

        if every is not None and every <= 0:
            raise ValueError("probe 'every' must be positive when provided")

        self.probe_enabled = True
        self.probe_script = script_path
        self.probe_out_root = out_root_path
        self.probe_truth_jsonl = truth_jsonl_path
        self.probe_truth_npy = truth_npy_path
        self.probe_truth_csv = truth_csv_path
        self.probe_dir = probe_dir_path
        self.probe_eval_dir = probe_eval_dir_path
        self.probe_tag_base = tag
        if every is not None:
            self.probe_every = every

    def setup_datamodule(self, datamodule: Datamodule) -> None:
        # Exract the necessary information from the datamodule
        self.standardize = datamodule.standardize
        self.fourier_transform = datamodule.fourier_transform
        self.feature_mean, self.feature_std = datamodule.feature_mean_and_std
        self.metric_collection = MetricCollection(
            metrics=self.metrics,
            original_samples=datamodule.X_train,
            include_baselines=False,
        )
        self.datamodule_initialized = True

    def on_train_start(self, trainer: pl.Trainer, pl_module: ScoreModule) -> None:
        # Initialize the sampler with the score model
        self.sampler = DiffusionSampler(
            score_model=pl_module,
            sample_batch_size=self.sample_batch_size,
        )
        if self.probe_enabled:
            log_dir = getattr(trainer.logger, "log_dir", None)
            if log_dir is not None:
                self._run_dir = Path(log_dir).expanduser()
            else:
                self._run_dir = Path.cwd() / "lightning_logs"
        else:
            self._run_dir = None

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        max_epochs = getattr(trainer, "max_epochs", None)
        is_last_epoch = max_epochs is not None and (trainer.current_epoch + 1 == max_epochs)
        should_sample = (
            trainer.current_epoch % self.every_n_epochs == 0 or is_last_epoch
        )
        if not should_sample:
            return

        X = self.sample()

        # Compute metrics
        results = self.metric_collection(X)

        # Add a metrics/ suffix to the keys in results
        results = {f"metrics/{key}": value for key, value in results.items()}

        # Log metrics
        pl_module.log_dict(results, on_step=False, on_epoch=True)
        self._maybe_run_probe(trainer, X)

    def _maybe_run_probe(self, trainer: pl.Trainer, samples_tensor: torch.Tensor) -> None:
        if not self.probe_enabled or self.probe_script is None:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.probe_every != 0 and (trainer.max_epochs is None or epoch != getattr(trainer, "max_epochs", None)):
            return

        run_dir = self._run_dir or Path.cwd() / "lightning_logs"
        epoch_dir = run_dir / f"probe_epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        samples_np = samples_tensor.detach().cpu().numpy()
        np.save(epoch_dir / "samples.npy", samples_np)

        pred_len = samples_np.shape[1]
        num_assets = samples_np.shape[2]
        num_samples = samples_np.shape[0]

        probe_out = self.probe_out_root or (run_dir.parent / "covcmp_fourier")
        cmd = [
            sys.executable,
            str(self.probe_script),
            "--fdiff_logs", str(epoch_dir),
            "--outdir", str(probe_out),
            "--pred_len", str(pred_len),
            "--num_assets", str(num_assets),
            "--num_samples", str(num_samples),
            "--timing_sample", str(self._last_sample_elapsed),
            "--probe_enable",
            "--probe_epochs", str(epoch),
        ]
        if self.probe_truth_jsonl:
            cmd.extend(["--truth_jsonl", str(self.probe_truth_jsonl)])
        if self.probe_truth_npy:
            cmd.extend(["--truth_npy", str(self.probe_truth_npy)])
        if self.probe_truth_csv:
            cmd.extend(["--truth_csv", str(self.probe_truth_csv)])
        if self.probe_dir:
            cmd.extend(["--probe_dir", str(self.probe_dir)])
        if self.probe_eval_dir:
            cmd.extend(["--probe_eval_dir", str(self.probe_eval_dir)])
        if self.probe_tag_base:
            cmd.extend(["--probe_tag", f"{self.probe_tag_base}_epoch{epoch:04d}"])

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[probe] cov_compare_FourierDiffusion 执行失败: {exc}")
        except FileNotFoundError as exc:
            print(f"[probe] cov_compare_FourierDiffusion 不可用: {exc}")
        except Exception as exc:
            print(f"[probe] cov_compare_FourierDiffusion 出现异常: {exc}")


    def sample(self) -> torch.Tensor:
        # Check that the datamodule is initialized
        assert self.datamodule_initialized, (
            "The datamodule has not been initialized. "
            "Please call `setup_datamodule` before sampling."
        )

        start = time.perf_counter()
        X = self.sampler.sample(
            num_samples=self.num_samples,
            num_diffusion_steps=self.num_diffusion_steps,
        )
        self._last_sample_elapsed = time.perf_counter() - start

        if self.standardize:
            X = X * self.feature_std + self.feature_mean

        if self.fourier_transform:
            X = idft(X)
        assert isinstance(X, torch.Tensor)
        return X
