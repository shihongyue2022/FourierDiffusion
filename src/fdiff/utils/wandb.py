# external/FourierDiffusion/src/fdiff/utils/wandb.py
"""
Safer W&B initializer for FourierDiffusion.

功能：
- 支持用环境变量一键禁用 W&B：WANDB_DISABLED=true|1|yes
- 支持用环境变量覆盖 entity / project：WANDB_ENTITY / WANDB_PROJECT
- 如果联网失败（例如 403 权限问题），自动降级 offline，不中断训练
- 返回一个 run_id（线上/离线都返回），供外层记录

用法：保持原来的调用不变
    from fdiff.utils.wandb import maybe_initialize_wandb
    run_id = maybe_initialize_wandb(cfg)

作者：ChatGPT（给 shi）
"""

from __future__ import annotations
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

# --- 尝试导入 wandb；若失败则给出降级 stub ---
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception as _e:
    _WANDB_AVAILABLE = False
    class _WandbStub:  # 极简 stub，保证模块可用
        def init(self, *args, **kwargs):
            return None
    wandb = _WandbStub()  # type: ignore


def _now_run_id(prefix: str = "offline") -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}"


def _as_bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if not v:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _flatten_cfg(cfg: Any) -> Dict[str, Any]:
    """
    把 Hydra/OmegaConf 或普通 dict 展平为 k->v 字典，用于 wandb.config。
    不追求严格扁平化，只需可序列化；复杂对象用 str()。
    """
    # 优先尝试 OmegaConf
    try:
        from omegaconf import OmegaConf  # type: ignore
        if OmegaConf.is_config(cfg):
            try:
                # to_container(..., resolve=True) 会把嵌套结构转为 Python 基本容器
                c = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
                return _flatten_dict_like(c)  # type: ignore
            except Exception:
                pass
    except Exception:
        pass

    # 普通 dict / 其他对象
    if isinstance(cfg, dict):
        return _flatten_dict_like(cfg)
    # 兜底：直接转字符串
    return {"cfg": str(cfg)}


def _flatten_dict_like(d: Dict[str, Any], parent: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else str(k)
        if isinstance(v, dict):
            flat.update(_flatten_dict_like(v, key))
        elif isinstance(v, (list, tuple)):
            # 列表直接字符串化，避免过大/不可序列化
            flat[key] = str(v)
        else:
            # 基本类型/可序列化对象
            try:
                _ = str(v) if v is None else v  # 尝试不改变基本类型
                flat[key] = v
            except Exception:
                flat[key] = str(v)
    return flat


def maybe_initialize_wandb(cfg: Any) -> str:
    """
    初始化（或跳过）W&B，并返回一个 run_id。
    - 若设置 WANDB_DISABLED=true/1/yes，则完全跳过 W&B，返回离线 run_id。
    - 若环境里提供 WANDB_ENTITY / WANDB_PROJECT，则用于覆盖默认配置。
    - 若初始化失败（例如 403），自动降级 offline，不中断训练。
    """
    # 1) 显式禁用
    if _as_bool_env("WANDB_DISABLED", False):
        rid = _now_run_id("offline")
        print(f"[wandb.py] WANDB_DISABLED is set; skip wandb.init(). run_id={rid}", file=sys.stderr)
        return rid

    # 2) 若模块不可用，直接离线
    if not _WANDB_AVAILABLE:
        rid = _now_run_id("offline")
        print(f"[wandb.py] wandb not available; falling back to offline. run_id={rid}", file=sys.stderr)
        return rid

    # 3) 组装 entity/project；默认不强制 entity，避免 'fdiff' 导致 403
    entity = os.getenv("WANDB_ENTITY", None)           # 如：shihongyue2022-tokyo
    project = os.getenv("WANDB_PROJECT", "FourierDiffusion")

    # 4) 展平 cfg 作为 config
    cfg_flat: Dict[str, Any] = {}
    try:
        cfg_flat = _flatten_cfg(cfg)
    except Exception as e:
        print(f"[wandb.py] flatten cfg failed: {e}", file=sys.stderr)
        cfg_flat = {"cfg": str(cfg)}

    # 5) 尝试联网初始化；若失败自动降级 offline
    try:
        kwargs = dict(project=project, config=cfg_flat)
        if entity:  # 仅当你设置了 WANDB_ENTITY 时才传，避免使用硬编码的 'fdiff'
            kwargs["entity"] = entity
        run = wandb.init(**kwargs)  # type: ignore
        if run is None:
            rid = _now_run_id("offline")
            print(f"[wandb.py] wandb.init() returned None; use offline run_id={rid}", file=sys.stderr)
            return rid
        rid = getattr(run, "id", None) or _now_run_id("wb")
        print(f"[wandb.py] wandb online. entity={entity or '(default)'} project={project} run_id={rid}")
        return rid
    except Exception as e:
        # 常见：权限 403 / 无网络等
        rid = _now_run_id("offline")
        print(f"[wandb.py] wandb.init() failed: {e}\n"
              f"           Falling back to offline. run_id={rid}", file=sys.stderr)
        return rid

