from __future__ import annotations
import logging
import os
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Any, Optional, List, Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from fdiff.utils.dataclasses import collate_batch
from fdiff.utils.fourier import dft, localization_metrics, smooth_frequency
from fdiff.utils.preprocessing import (
    droughts_preprocess,
    mimic_preprocess,
    nasa_preprocess,
    nasdaq_preprocess,
)


# -----------------------------
# Generic dataset for diffusion
# -----------------------------
class DiffusionDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        fourier_transform: bool = False,
        standardize: bool = False,
        X_ref: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if fourier_transform:
            X = dft(X).detach()
        self.X = X
        self.y = y
        self.standardize = standardize
        if X_ref is None:
            X_ref = X
        elif fourier_transform:
            X_ref = dft(X_ref).detach()
        assert isinstance(X_ref, torch.Tensor)
        self.feature_mean = X_ref.mean(dim=0)
        self.feature_std = X_ref.std(dim=0)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data: dict[str, torch.Tensor] = {}
        data["X"] = self.X[index]
        if self.standardize:
            data["X"] = (data["X"] - self.feature_mean) / self.feature_std
        if self.y is not None:
            data["y"] = self.y[index]
        return data


# -----------------------------
# Base Lightning DataModule
# -----------------------------
class Datamodule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        data_dir: Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        fourier_transform: bool = False,
        standardize: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.data_dir = data_dir / self.dataset_name
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.fourier_transform = fourier_transform
        self.standardize = standardize
        self.X_train = torch.Tensor()
        self.y_train: Optional[torch.Tensor] = None
        self.X_test = torch.Tensor()
        self.y_test: Optional[torch.Tensor] = None

    def prepare_data(self) -> None:
        if not self.data_dir.exists():
            logging.info(f"Downloading {self.dataset_name} dataset in {self.data_dir}.")
            os.makedirs(self.data_dir)
            self.download_data()

    @abstractmethod
    def download_data(self) -> None:
        ...

    def train_dataloader(self) -> DataLoader:
        train_set = DiffusionDataset(
            X=self.X_train,
            y=self.y_train,
            fourier_transform=self.fourier_transform,
            standardize=self.standardize,
        )
        return DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, collate_fn=collate_batch
        )

    def test_dataloader(self) -> DataLoader:
        test_set = DiffusionDataset(
            X=self.X_test, y=self.y_test, fourier_transform=self.fourier_transform
        )
        return DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, collate_fn=collate_batch
        )

    def val_dataloader(self) -> DataLoader:
        test_set = DiffusionDataset(
            X=self.X_test,
            y=self.y_test,
            fourier_transform=self.fourier_transform,
            standardize=self.standardize,
            X_ref=self.X_train,
        )
        return DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, collate_fn=collate_batch
        )

    @abstractproperty
    def dataset_name(self) -> str: ...

    @property
    def dataset_parameters(self) -> dict[str, Any]:
        return {
            "n_channels": self.X_train.size(2),
            "max_len": self.X_train.size(1),
            "num_training_steps": len(self.train_dataloader()),
        }

    @property
    def feature_mean_and_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        train_set = DiffusionDataset(
            X=self.X_train,
            y=self.y_train,
            fourier_transform=self.fourier_transform,
            standardize=self.standardize,
        )
        return train_set.feature_mean, train_set.feature_std


# -----------------------------
# ECG
# -----------------------------
class ECGDatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        fourier_transform: bool = False,
        standardize: bool = False,
        subsample_localization: bool = False,
        smooth_frequency: bool = False,
        smoother_width: float = 0.0,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=fourier_transform,
            standardize=standardize,
        )
        self.subsample_localization = subsample_localization
        self.smooth_frequency = smooth_frequency
        self.smoother_width = smoother_width

    def setup(self, stage: str = "fit") -> None:
        path_train = self.data_dir / "mitbih_train.csv"
        path_test = self.data_dir / "mitbih_test.csv"
        df_train = pd.read_csv(path_train)
        X_train = df_train.iloc[:, :187].values
        y_train = df_train.iloc[:, 187].values
        df_test = pd.read_csv(path_test)
        X_test = df_test.iloc[:, :187].values
        y_test = df_test.iloc[:, 187].values

        self.X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
        self.y_test = torch.tensor(y_test, dtype=torch.long)

        if self.subsample_localization:
            X_loc, X_spec_loc = localization_metrics(self.X_train)
            loc_score = X_loc / X_spec_loc
            idx_ranking = torch.argsort(loc_score, descending=False)
            self.X_train = self.X_train[idx_ranking[:1000]]
            self.y_train = self.y_train[idx_ranking[:1000]]
            X_loc, X_spec_loc = localization_metrics(self.X_train)
            logging.info("Subsampling the training set based on localization metrics.")
            logging.info(f"New time delocalization: {X_loc.mean().item():.3g}")
            logging.info(
                f"New frequency delocalization: {X_spec_loc.mean().item():.3g}"
            )

        if self.smooth_frequency and self.smoother_width > 0.0:
            self.X_train = smooth_frequency(self.X_train, sigma=self.smoother_width)
            self.X_test = smooth_frequency(self.X_test, sigma=self.smoother_width)
            logging.info("Smoothing the frequency domain of the data.")
            X_loc, X_spec_loc = localization_metrics(self.X_train)
            logging.info(f"New time delocalization: {X_loc.mean().item():.3g}")
            logging.info(
                f"New frequency delocalization: {X_spec_loc.mean().item():.3g}"
            )

    def download_data(self) -> None:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "shayanfazeli/heartbeat", path=self.data_dir, unzip=True
        )

    @property
    def dataset_name(self) -> str:
        return "ecg"


# -----------------------------
# Synthetic
# -----------------------------
class SyntheticDatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        fourier_transform: bool = False,
        standardize: bool = False,
        max_len: int = 100,
        num_samples: int = 1000,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=fourier_transform,
            standardize=standardize,
        )
        self.max_len = max_len
        self.num_samples = num_samples

    def setup(self, stage: str = "fit") -> None:
        path_train = self.data_dir / "train.csv"
        path_test = self.data_dir / "test.csv"
        df_train = pd.read_csv(path_train, header=None)
        X_train = df_train.values
        df_test = pd.read_csv(path_test, header=None)
        X_test = df_test.values

        self.X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
        self.y_train = None
        self.X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
        self.y_test = None

    def download_data(self) -> None:
        n_generated = 2 * self.num_samples
        phase = np.random.normal(size=(n_generated)).reshape(-1, 1)
        frequency = np.random.beta(a=2, b=2, size=(n_generated)).reshape(-1, 1)
        timesteps = np.arange(self.max_len)
        X = np.sin(timesteps * frequency + phase)
        X_train = X[: self.num_samples]
        X_test = X[self.num_samples :]

        df_train = pd.DataFrame(X_train)
        df_test = pd.DataFrame(X_test)
        df_train.to_csv(self.data_dir / "train.csv", index=False, header=False)
        df_test.to_csv(self.data_dir / "test.csv", index=False, header=False)

    @property
    def dataset_name(self) -> str:
        return "synthetic"


# -----------------------------
# MIMIC-III
# -----------------------------
class MIMICIIIDatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        fourier_transform: bool = False,
        standardize: bool = False,
        n_feats: int = 40,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=fourier_transform,
            standardize=standardize,
        )
        self.n_feats = n_feats

    def setup(self, stage: str = "fit") -> None:
        if (
            not (self.data_dir / "X_train.pt").exists()
            or not (self.data_dir / "X_test.pt").exists()
        ):
            logging.info(
                f"Preprocessed tensors for {self.dataset_name} not found. "
                f"Now running the preprocessing pipeline."
            )
            mimic_preprocess(data_dir=self.data_dir, random_seed=self.random_seed)
            logging.info(
                f"Preprocessing pipeline finished, tensors saved in {self.data_dir}."
            )

        self.X_train = torch.load(self.data_dir / "X_train.pt")
        self.X_test = torch.load(self.data_dir / "X_test.pt")
        assert isinstance(self.X_train, torch.Tensor)
        assert isinstance(self.X_test, torch.Tensor)

        top_feats = torch.argsort(self.X_train.std(0).mean(0), descending=True)[
            : self.n_feats
        ]
        self.X_train = self.X_train[:, :, top_feats]
        self.X_test = self.X_test[:, :, top_feats]

    def download_data(self) -> None:
        dataset_path = self.data_dir / "all_hourly_data.h5"
        assert dataset_path.exists(), (
            f"Dataset {dataset_path} does not exist. "
            "Because MIMIC-III is a restricted dataset, you need to download it yourself. "
            "Our implementation relies on the default MIMIC-Extract preprocessed version of the dataset. "
            "Please follow the instruction on https://github.com/MLforHealth/MIMIC_Extract/tree/master."
        )

    @property
    def dataset_name(self) -> str:
        return "mimiciii"


# -----------------------------
# NASDAQ
# -----------------------------
class NASDAQDatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        fourier_transform: bool = False,
        standardize: bool = False,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=fourier_transform,
            standardize=standardize,
        )

    def setup(self, stage: str = "fit") -> None:
        if (
            not (self.data_dir / "X_train.pt").exists()
            or not (self.data_dir / "X_test.pt").exists()
        ):
            logging.info(
                f"Preprocessed tensors for {self.dataset_name} not found. "
                f"Now running the preprocessing pipeline."
            )
            nasdaq_preprocess(data_dir=self.data_dir, random_seed=self.random_seed)
            logging.info(
                f"Preprocessing pipeline finished, tensors saved in {self.data_dir}."
            )

        self.X_train = torch.load(self.data_dir / "X_train.pt")
        self.X_test = torch.load(self.data_dir / "X_test.pt")
        assert isinstance(self.X_train, torch.Tensor)
        assert isinstance(self.X_test, torch.Tensor)
        assert self.X_train.shape[1:] == self.X_test.shape[1:] == (252, 6)

        self.X_train = self.X_train[:, :, :-1]
        self.X_test = self.X_test[:, :, :-1]

    def download_data(self) -> None:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "jacksoncrow/stock-market-dataset", path=self.data_dir, unzip=True
        )

    @property
    def dataset_name(self) -> str:
        return "nasdaq"


# -----------------------------
# NASA
# -----------------------------
class NASADatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        fourier_transform: bool = False,
        standardize: bool = False,
        subdataset: str = "charge",
        remove_outlier_feature: bool = True,
    ) -> None:
        self.subdataset = subdataset
        self.remove_outlier_feature = remove_outlier_feature

        super().__init__(
            data_dir=data_dir,
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=fourier_transform,
            standardize=standardize,
        )

    def setup(self, stage: str = "fit") -> None:
        if (
            not (self.data_dir / self.subdataset / "X_train.pt").exists()
            or not (self.data_dir / self.subdataset / "X_test.pt").exists()
        ):
            logging.info(
                f"Preprocessed tensors for {self.dataset_name}_{self.subdataset} not found. "
                f"Now running the preprocessing pipeline."
            )
            nasa_preprocess(
                data_dir=self.data_dir,
                subdataset=self.subdataset,
                random_seed=self.random_seed,
            )
            logging.info(
                f"Preprocessing pipeline finished, tensors saved in {self.data_dir}."
            )

        self.X_train = torch.load(self.data_dir / self.subdataset / "X_train.pt")
        self.X_test = torch.load(self.data_dir / self.subdataset / "X_test.pt")

        if self.remove_outlier_feature and self.subdataset == "charge":
            self.X_train = self.X_train[:, ::2, [0, 1, 3, 4]]
            self.X_test = self.X_test[:, ::2, [0, 1, 3, 4]]

            assert self.X_train.shape[2] == self.X_test.shape[2] == 4
            assert self.X_train.shape[1] == 251
            assert self.X_test.shape[1] == 251
        assert isinstance(self.X_train, torch.Tensor)
        assert isinstance(self.X_test, torch.Tensor)

    def download_data(self) -> None:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "patrickfleith/nasa-battery-dataset", path=self.data_dir, unzip=True
        )

    @property
    def dataset_name(self) -> str:
        return "nasa"


# -----------------------------
# US Droughts
# -----------------------------
class USDroughtsDatamodule(Datamodule):
    def __init__(
        self,
        data_dir: Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        fourier_transform: bool = False,
        standardize: bool = False,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            random_seed=random_seed,
            batch_size=batch_size,
            fourier_transform=fourier_transform,
            standardize=standardize,
        )

    def setup(self, stage: str = "fit") -> None:
        if (
            not (self.data_dir / "X_train.pt").exists()
            or not (self.data_dir / "X_test.pt").exists()
        ):
            logging.info(
                f"Preprocessed tensors for {self.dataset_name} not found. "
                f"Now running the preprocessing pipeline."
            )
            droughts_preprocess(data_dir=self.data_dir, random_seed=self.random_seed)

            logging.info(
                f"Preprocessing pipeline finished, tensors saved in {self.data_dir}."
            )

        self.X_train: torch.Tensor = torch.load(self.data_dir / "X_train.pt")
        self.X_test: torch.Tensor = torch.load(self.data_dir / "X_test.pt")

        feats = [i for i in range(self.X_train.shape[2]) if i not in {4, 5, 6, 7, 9}]
        self.X_train = self.X_train[:, :, feats]
        self.X_test = self.X_test[:, :, feats]

        assert isinstance(self.X_train, torch.Tensor)
        assert isinstance(self.X_test, torch.Tensor)
        assert self.X_train.shape[1] % 365 == self.X_test.shape[1] % 365 == 0
        assert self.X_train.shape[2] == self.X_test.shape[2] == len(feats)

    def download_data(self) -> None:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "cdminix/us-drought-meteorological-data", path=self.data_dir, unzip=True
        )

    @property
    def dataset_name(self) -> str:
        return "droughts"


# ======================================================================
# GluonTS JSON 数据目录（train/test -> *.json 或 data.json(.gz)）
# ======================================================================

def _fdj_resolve_split_file(root: Path, split: str) -> Path:
    cands = [
        root / f"{split}.json",
        root / split / f"{split}.json",
        root / split / "data.json",
        root / split / "data.json.gz",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f"[GluonTSJsonDatamodule] 找不到 {split} 集：期望之一 {cands}")


def _fdj_iter_jsonl(path: Path):
    if path.suffix == ".gz":
        import gzip, json
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        import json
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _fdj_load_gluonts_like(path: Path) -> np.ndarray:
    series: List[List[float]] = []
    for obj in _fdj_iter_jsonl(path):
        tgt = obj.get("target", None)
        if tgt is None:
            raise ValueError(f"{path}: 缺少 'target' 字段")
        series.append(list(map(float, tgt)))

    if not series:
        raise ValueError(f"{path}: 空数据")

    min_len = min(len(s) for s in series)
    series = [s[-min_len:] for s in series]
    arr = np.stack(series, axis=1)  # (T, A)
    return arr.astype(np.float32, copy=False)


def _fdj_build_windows(X: np.ndarray, L: int, stride: int) -> np.ndarray:
    T, A = X.shape
    if T < L:
        raise ValueError(f"时间长度 T={T} < 窗口长度 L={L}")
    idx = []
    t = 0
    while t + L <= T:
        idx.append((t, t + L))
        t += stride
    return np.stack([X[s:e, :] for (s, e) in idx], axis=0).astype(np.float32, copy=False)


class _FDJWindowedDataset(Dataset):
    def __init__(self, windows: np.ndarray):
        self.windows = windows  # (N, L, A) float32

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        x = self.windows[i]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)  # (L, A)
        return {"X": x}


class GluonTSJsonDatamodule(pl.LightningDataModule):
    """
    读取 GluonTS 数据目录（含 train/test 子目录或同名 json 文件）。
    自动识别 train.json / train/data.json(.gz) 与 test 同理。
    - 标准化：按资产（用 train 的均值/方差）
    - 输出：训练/验证/测试 DataLoader（样本为窗口 (L, A)）
    - 兼容 SamplingCallback：提供 X_train/X_test、feature_mean/feature_std
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        window_length: int = 360,
        pred_len: int = 30,
        stride: int = 1,
        val_ratio: float = 0.1,
        standardize: bool = True,
        fourier_transform: bool = False,
        num_workers: int = 4,
        pin_memory: bool = True,
        jsonl_train: Optional[str] = None,
        jsonl_test: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = int(batch_size)
        self.window_length = int(window_length)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.val_ratio = float(val_ratio)
        self.standardize = bool(standardize)
        self.fourier_transform = bool(fourier_transform)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

        self.jsonl_train = Path(jsonl_train) if jsonl_train else None
        self.jsonl_test = Path(jsonl_test) if jsonl_test else None

        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

        self.ds_train: Optional[Dataset] = None
        self.ds_val: Optional[Dataset] = None
        self.ds_test: Optional[Dataset] = None

        # 为 SamplingCallback/metrics 兼容准备：
        self.X_train: Optional[torch.Tensor] = None  # (N_tr, L, A)
        self.X_test: Optional[torch.Tensor] = None   # (N_te, L, A)
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None

        self.num_assets_: Optional[int] = None
        self.input_length_: Optional[int] = None

    @property
    def dataset_name(self) -> str:
        return "gluonts_json"

    def prepare_data(self):
        root = self.data_dir
        if not root.exists():
            raise FileNotFoundError(f"data_dir 不存在：{root}")
        if self.jsonl_train is None:
            self.jsonl_train = _fdj_resolve_split_file(root, "train")
        if self.jsonl_test is None:
            try:
                self.jsonl_test = _fdj_resolve_split_file(root, "test")
            except FileNotFoundError:
                self.jsonl_test = None

    def setup(self, stage: Optional[str] = None):
        assert self.jsonl_train is not None
        Xtr = _fdj_load_gluonts_like(self.jsonl_train)  # (Ttr, A)
        Atr = Xtr.shape[1]
        Xt = None
        if self.jsonl_test is not None:
            Xt = _fdj_load_gluonts_like(self.jsonl_test)  # (Tte, A)
            if Xt.shape[1] != Atr:
                raise ValueError(f"train/test 资产数不一致: {Atr} vs {Xt.shape[1]}")

        # 标准化
        if self.standardize:
            mu = Xtr.mean(axis=0)
            sigma = Xtr.std(axis=0, ddof=1)
            sigma[sigma < 1e-8] = 1.0
            self._mean, self._std = mu, sigma
            Xtr = (Xtr - mu) / sigma
            if Xt is not None:
                Xt = (Xt - mu) / sigma
            # Torch 版缓存（给回调用）
            self.feature_mean = torch.from_numpy(mu.astype(np.float32))
            self.feature_std = torch.from_numpy(sigma.astype(np.float32))
        else:
            self.feature_mean = None
            self.feature_std = None

        # 滑窗并构建 Dataset
        win_tr = _fdj_build_windows(Xtr, self.window_length, self.stride)  # (Ntr,L,A)
        ntr = win_tr.shape[0]
        nval = max(1, int(round(ntr * self.val_ratio)))
        ntr_keep = max(1, ntr - nval)
        self.ds_train = _FDJWindowedDataset(win_tr[:ntr_keep])
        self.ds_val = _FDJWindowedDataset(win_tr[ntr_keep:]) if nval > 0 else _FDJWindowedDataset(win_tr[:ntr_keep])

        # 兼容 SamplingCallback：提供 X_train 窗口张量
        self.X_train = torch.from_numpy(win_tr[:ntr_keep])  # (N_tr, L, A)

        if Xt is not None:
            win_te = _fdj_build_windows(Xt, self.window_length, self.stride)
            self.ds_test = _FDJWindowedDataset(win_te)
            self.X_test = torch.from_numpy(win_te)          # (N_te, L, A)
        else:
            self.ds_test = None
            self.X_test = None

        self.num_assets_ = Atr
        self.input_length_ = self.window_length

    # ---- dataloaders ----
    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=collate_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_batch,
        )

    def test_dataloader(self):
        if self.ds_test is None:
            return None
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_batch,
        )

    # ---- helpers for get_training_params / callbacks ----
    @property
    def dataset_parameters(self) -> dict:
        if self.ds_train is None:
            raise RuntimeError("Call setup() before accessing dataset_parameters")
        sample = self.ds_train[0]["X"]
        n_channels = int(sample.shape[-1])
        max_len = int(self.window_length)
        num_training_steps = len(self.train_dataloader())
        return {
            "n_channels": n_channels,
            "max_len": max_len,
            "num_training_steps": num_training_steps,
        }

    @property
    def feature_mean_and_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feature_mean is not None and self.feature_std is not None:
            return self.feature_mean, self.feature_std
        # 未标准化：返回 0/1
        sample = self.ds_train[0]["X"]
        a = int(sample.shape[-1])
        return torch.zeros(a, dtype=torch.float32), torch.ones(a, dtype=torch.float32)


# 若文件里维护了 __all__，把新类名加进去；若没有 __all__ 则无事发生
try:
    __all__ = list(__all__)  # type: ignore
    if "GluonTSJsonDatamodule" not in __all__:
        __all__.append("GluonTSJsonDatamodule")
except NameError:
    __all__ = ["GluonTSJsonDatamodule"]
