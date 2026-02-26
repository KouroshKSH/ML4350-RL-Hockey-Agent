import os
import glob
import yaml
import numpy as np
import torch

from .normalization import RunningMeanStd
from .networks import Actor
from .tqc_agent import GaussianPolicy  # used by SAC and TQC


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_obs_norm_npz(norm_path: str, state_dim: int) -> RunningMeanStd:
    data = np.load(norm_path, allow_pickle=True)
    # np.load returns an NpzFile that supports .get()
    clip = float(data.get("clip", 5.0))
    rms = RunningMeanStd(shape=(state_dim,), clip=clip)
    rms.mean = np.array(data["mean"], dtype=np.float64)
    rms.var = np.array(data["var"], dtype=np.float64)
    rms.count = float(data["count"])
    return rms


def actor_spec_from_config(algo: str, run_cfg: dict, state_dim: int, action_dim: int):
    """
    returns the tuple actor_cls, actor_kwargs, has_sample
    it must match how actors were constructed when they were being saved
    """
    algo = str(algo).upper()

    if algo == "TD3":
        td3_cfg = run_cfg.get("td3", {}) or {}
        hidden = int(td3_cfg.get("actor_hidden_dim", 256))
        return Actor, {"state_dim": state_dim, "action_dim": action_dim, "hidden_dim": hidden}, False

    if algo == "SAC":
        sac_cfg = run_cfg.get("sac", {}) or {}
        hidden = int(sac_cfg.get("actor_hidden_dim", 256))
        return GaussianPolicy, {"state_dim": state_dim, "action_dim": action_dim, "hidden_dim": hidden}, True

    if algo == "TQC":
        tqc_cfg = run_cfg.get("tqc", {}) or {}
        hidden = int(tqc_cfg.get("actor_hidden_dim", 256))
        return GaussianPolicy, {"state_dim": state_dim, "action_dim": action_dim, "hidden_dim": hidden}, True

    raise ValueError(f"Unsupported external algo '{algo}' in this loader.")


def discover_external_models(cfg: dict, state_dim: int, action_dim: int):
    """
    check the results folders and return a list of entries:
      {
        "algo", "run_dir", "actor_path", "norm_path",
        "actor_cls", "actor_kwargs", "has_sample",
        "state_dict", "obs_norm", "strength"
      }
    """
    ext = cfg.get("external_pool", {}) or {}
    if not bool(ext.get("enabled", False)):
        return []

    # get the project root from this file location:
    # project_root/agent/external_models.py
    inferred_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    results_root_raw = str(ext.get("results_root", "results"))
    results_root_raw = os.path.expanduser(results_root_raw)
    if os.path.isabs(results_root_raw):
        results_root = os.path.abspath(results_root_raw)
    else:
        results_root = os.path.abspath(os.path.join(inferred_project_root, results_root_raw))

    project_root = os.path.dirname(results_root)

    def _resolve_manifest_path(p):
        if not p:
            return None
        p = os.path.expanduser(str(p))
        return p if os.path.isabs(p) else os.path.join(project_root, p)

    env_name = str(ext.get("env_name", "hockey"))
    threshold = float(ext.get("score_threshold", 0.89))

    algos_allow = ext.get("algos", None)
    if algos_allow:
        algos_allow = set([str(a).upper() for a in algos_allow])

    include_segments = ext.get("include_segments", ["best", "mid", "late"])
    include_segments = set([str(s).lower() for s in include_segments])

    max_models = int(ext.get("max_models", 999999))
    max_per_run = int(ext.get("max_per_run", 999999))
    require_norm = bool(ext.get("require_norm", True))

    # results/<ALGO>/<env_name>/<RUN_NAME>/best_models_manifest.yaml
    manifest_glob = os.path.join(results_root, "*", env_name, "*", "best_models_manifest.yaml")
    manifests = sorted(glob.glob(manifest_glob))

    entries_out = []
    seen_actor_paths = set()

    for man_path in manifests:
        run_dir = os.path.dirname(man_path)
        run_cfg_path = os.path.join(run_dir, "config.yaml")
        if not os.path.exists(run_cfg_path):
            continue

        man = _load_yaml(man_path)
        algo = str(man.get("algo", "")).upper()
        if algos_allow and algo not in algos_allow:
            continue

        run_cfg = _load_yaml(run_cfg_path)

        per_run_count = 0
        for e in (man.get("entries", []) or []):
            seg = str(e.get("segment", "")).lower()
            if seg not in include_segments:
                continue

            best_score = e.get("best_score", None)
            if best_score is None:
                continue
            try:
                best_score = float(best_score)
            except Exception:
                continue
            if best_score < threshold:
                continue

            actor_path = _resolve_manifest_path(e.get("actor_path", None))
            norm_path = _resolve_manifest_path(e.get("norm_path", None))

            if not actor_path or not os.path.exists(actor_path):
                continue
            if actor_path in seen_actor_paths:
                continue

            if require_norm:
                if (not norm_path) or (not os.path.exists(norm_path)):
                    continue

            # given the run config, build the actor's spec
            actor_cls, actor_kwargs, has_sample = actor_spec_from_config(algo, run_cfg, state_dim, action_dim)

            # for loading the weights
            sd = torch.load(actor_path, map_location="cpu")
            if not isinstance(sd, dict):
                continue

            # for normalizing the state dict to CPU tensors
            state_dict = {}
            for k, v in sd.items():
                if torch.is_tensor(v):
                    state_dict[k] = v.detach().cpu()
                else:
                    state_dict[k] = v

            # loads the observation normalizations per model
            obs_norm = None
            if norm_path and os.path.exists(norm_path):
                try:
                    obs_norm = load_obs_norm_npz(norm_path, state_dim=state_dim)
                except Exception:
                    if require_norm:
                        continue

            entries_out.append({
                "algo": algo,
                "run_dir": run_dir,
                "actor_path": actor_path,
                "norm_path": norm_path,
                "actor_cls": actor_cls,
                "actor_kwargs": actor_kwargs,
                "has_sample": has_sample,
                "state_dict": state_dict,
                "obs_norm": obs_norm,
                "strength": best_score,
            })
            seen_actor_paths.add(actor_path)

            per_run_count += 1
            if per_run_count >= max_per_run:
                break
            if len(entries_out) >= max_models:
                return entries_out

    return entries_out
