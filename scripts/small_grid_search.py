import os
import sys
import time
import glob
import csv
import yaml
import argparse
import subprocess
from copy import deepcopy
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PY = os.path.join(REPO_ROOT, "agent", "train.py")
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


# 2 presets per algo (base + 1 high-leverage variant) for gri search. I keep this small for now since we don't have time.
# But we can increase this and do more grid search if it seems we have time for more.
PRESETS = {
    "TQC": [
        ("base", {}),
        ("ramp_smooth", {"curriculum.self_play_ramp_episodes": 1200, "self_play.pfsp_temperature": 0.25}),
    ],
    "SAC": [
        ("base", {}),
        ("ramp_smooth", {"curriculum.self_play_ramp_episodes": 1200, "self_play.pfsp_temperature": 0.25}),
    ],
    "REDQ": [
        ("base", {}),
        ("utd_lower", {"updates_per_step": 1}),  # keep stable in nonstationary self-play
    ],
    "DROQ": [
        ("base", {}),
        ("utd_lower", {"updates_per_step": 1}),
    ],
    "CROSSQ": [
        ("base", {}),
        ("ramp_smooth", {"curriculum.self_play_ramp_episodes": 1200, "self_play.pfsp_temperature": 0.25}),
    ],
    "TD3": [
        ("base", {}),
        ("tau_up", {"tau": 0.02}),
    ],
    "DDPG": [
        ("base", {}),
        ("tau_up", {"tau": 0.02}),
    ],
}


def set_by_path(cfg: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: str, cfg: dict):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _rescale_int(x, scale, min_value=1):
    try:
        v = int(round(float(x) * float(scale)))
        return max(min_value, v)
    except Exception:
        return None


def rescale_schedule(cfg: dict, new_total_episodes: int) -> dict:
    """
    Compresses the whole curriculum so all phases still occur within new_total_episodes.
    Also rescales self-play/PSRO schedule knobs that depend on episode counts.
    """
    cfg = deepcopy(cfg)

    phases = cfg.get("phases", []) or []
    if not phases:
        cfg["episodes"] = int(new_total_episodes)
        return cfg

    try:
        old_total = int(phases[-1].get("until_episode", cfg.get("episodes", new_total_episodes)))
    except Exception:
        old_total = int(cfg.get("episodes", new_total_episodes))

    if old_total <= 0:
        old_total = int(cfg.get("episodes", new_total_episodes)) or new_total_episodes

    scale = float(new_total_episodes) / float(old_total)

    new_phases = []
    prev_end = 0
    for ph in phases:
        ph2 = dict(ph or {})
        old_end = ph2.get("until_episode", old_total)
        new_end = _rescale_int(old_end, scale, min_value=prev_end + 1)
        if new_end is None:
            continue
        new_end = min(new_end, new_total_episodes)
        ph2["until_episode"] = int(new_end)
        new_phases.append(ph2)
        prev_end = int(new_end)
        if prev_end >= new_total_episodes:
            break

    if not new_phases:
        cfg["episodes"] = int(new_total_episodes)
        return cfg

    new_phases[-1]["until_episode"] = int(new_total_episodes)
    cfg["phases"] = new_phases
    cfg["episodes"] = int(new_total_episodes)

    sp = cfg.get("self_play", {}) or {}
    if isinstance(sp, dict):
        if "snapshot_start_episode" in sp:
            sp["snapshot_start_episode"] = int(max(0, _rescale_int(sp["snapshot_start_episode"], scale, min_value=0) or 0))
        if "snapshot_interval_episodes" in sp:
            sp["snapshot_interval_episodes"] = int(max(1, _rescale_int(sp["snapshot_interval_episodes"], scale, min_value=1) or 1))
        cfg["self_play"] = sp

    cur = cfg.get("curriculum", {}) or {}
    if isinstance(cur, dict):
        if "self_play_ramp_episodes" in cur:
            cur["self_play_ramp_episodes"] = int(max(1, _rescale_int(cur["self_play_ramp_episodes"], scale, min_value=1) or 1))
        cfg["curriculum"] = cur

    psro = cfg.get("psro", {}) or {}
    if isinstance(psro, dict):
        if "start_episode" in psro:
            psro["start_episode"] = int(max(0, _rescale_int(psro["start_episode"], scale, min_value=0) or 0))
        if "update_interval_episodes" in psro:
            psro["update_interval_episodes"] = int(max(1, _rescale_int(psro["update_interval_episodes"], scale, min_value=1) or 1))
        cfg["psro"] = psro

    return cfg


def find_run_dir(algo: str, env: str, user_tag: str, seed: int) -> str | None:
    """
    Train.py writes run_dir:
      results/{algo}/{env}/{timestamp}_{user}_seed{seed}
    """
    base = os.path.join(RESULTS_DIR, algo, env)
    if not os.path.isdir(base):
        return None

    pattern = os.path.join(base, f"*_{user_tag}_seed{seed}")
    matches = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not matches:
        return None
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def read_manifest_score(run_dir: str) -> float | None:
    """
    Uses best_models_manifest.yaml written by train.py.
    """
    man = os.path.join(run_dir, "best_models_manifest.yaml")
    if not os.path.exists(man):
        return None

    with open(man, "r") as f:
        data = yaml.safe_load(f) or {}
    entries = data.get("entries", []) or []
    if not entries:
        return None

    for e in entries:
        if e.get("best_score") is None:
            continue
        if "phase4" in str(e.get("phase_name", "")).lower() and str(e.get("segment", "")) == "late":
            return float(e["best_score"])

    last_idx = max(int(e.get("phase_idx", 0)) for e in entries)
    late_last = [e for e in entries if int(e.get("phase_idx", -1)) == last_idx and str(e.get("segment", "")) == "late"]
    if late_last and late_last[0].get("best_score") is not None:
        return float(late_last[0]["best_score"])

    best = None
    for e in entries:
        s = e.get("best_score")
        if s is None:
            continue
        s = float(s)
        best = s if (best is None or s > best) else best
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="*", default=["TQC", "SAC", "REDQ", "DROQ", "CROSSQ", "TD3"])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", nargs="*", type=int, default=[43, 44, 45])
    ap.add_argument("--env", type=str, default="hockey")
    ap.add_argument("--user_prefix", type=str, default="sweep3000")

    # If set, export WANDB_MODE to subprocesses
    ap.add_argument("--wandb_mode", type=str, default=None, choices=[None, "online", "offline", "disabled"])

    ap.add_argument("--eval_interval", type=int, default=150)
    ap.add_argument("--eval_eps_per_opp", type=int, default=10)
    ap.add_argument("--eval_opponents", nargs="*", default=["weak", "strong", "self_play"])

    # Actual project will be:
    #   f"{base}_{ALGO}"
    ap.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Base wandb project name; actual per-algo project becomes BASE_<ALGO> (optional).",
    )

    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = os.path.join(REPO_ROOT, "sweeps", f"sweep_{stamp}")
    cfg_out_dir = os.path.join(sweep_root, "configs")
    os.makedirs(cfg_out_dir, exist_ok=True)

    summary = []
    best_by_algo = {}

    base_env = os.environ.copy()
    if args.wandb_mode is not None:
        base_env["WANDB_MODE"] = args.wandb_mode

    for algo in args.algos:
        algo_u = algo.upper()
        if algo_u not in PRESETS:
            print(f"[WARN] No presets defined for {algo_u}. Skipping.")
            continue

        base_cfg_path = os.path.join(CONFIG_DIR, f"{algo_u.lower()}_private.yaml")
        if not os.path.exists(base_cfg_path):
            print(f"[WARN] Missing config: {base_cfg_path}. Skipping {algo_u}.")
            continue

        base_cfg = load_yaml(base_cfg_path)
        base_cfg["algo"] = algo_u
        base_cfg["env"] = args.env

        # Decide per-algo W&B project name
        base_project = (
            str(args.wandb_project).strip()
            if args.wandb_project is not None
            else (str(base_cfg.get("wandb_project")).strip() if base_cfg.get("wandb_project") else None)
        )
        if not base_project:
            base_project = "RL-hockey"
        per_algo_project = f"{base_project}_{algo_u}"

        # Apply eval settings for sweep
        base_cfg.setdefault("eval", {})
        base_cfg["eval"]["enabled"] = True
        base_cfg["eval"]["interval_episodes"] = int(args.eval_interval)
        base_cfg["eval"]["episodes_per_opponent"] = int(args.eval_eps_per_opp)
        base_cfg["eval"]["opponents"] = list(args.eval_opponents)

        # Reduce disk noise during sweep
        base_cfg["checkpoint_interval_episodes"] = 0

        presets = PRESETS[algo_u]

        for preset_name, overrides in presets:
            for seed in args.seeds:
                cfg = deepcopy(base_cfg)
                cfg = rescale_schedule(cfg, int(args.episodes))
                cfg["seed"] = int(seed)

                # Force per-algo project here so every run under this algo goes to its own W&B project
                cfg["wandb_project"] = per_algo_project

                # Apply preset overrides after rescale (so they win)
                for k, v in (overrides or {}).items():
                    set_by_path(cfg, k, v)

                uniq = f"{int(time.time() * 1000) % 100000}"
                user_tag = f"{args.user_prefix}_{preset_name}_s{seed}_{uniq}"
                cfg["user"] = user_tag

                cfg_path = os.path.join(cfg_out_dir, f"{algo_u}__{preset_name}__seed{seed}.yaml")
                dump_yaml(cfg_path, cfg)

                cmd = [sys.executable, TRAIN_PY, "--config", cfg_path, "--seed", str(seed), "--user", user_tag]
                print(f"\n[RUN] {algo_u} preset={preset_name} seed={seed} episodes={args.episodes}")
                print(f"      wandb_project={per_algo_project}")
                print("      " + " ".join(cmd))

                t0 = time.time()
                p = subprocess.run(cmd, cwd=REPO_ROOT, env=base_env)
                dt = time.time() - t0

                if p.returncode != 0:
                    row = {
                        "algo": algo_u,
                        "preset": preset_name,
                        "seed": seed,
                        "episodes": int(args.episodes),
                        "score": None,
                        "status": "FAILED",
                        "seconds": round(dt, 1),
                        "run_dir": None,
                        "config": cfg_path,
                        "overrides": overrides,
                        "wandb_project": per_algo_project,
                    }
                    summary.append(row)
                    continue

                run_dir = find_run_dir(algo_u, args.env, user_tag, int(seed))
                score = read_manifest_score(run_dir) if run_dir else None

                row = {
                    "algo": algo_u,
                    "preset": preset_name,
                    "seed": seed,
                    "episodes": int(args.episodes),
                    "score": score,
                    "status": "OK" if score is not None else "NO_SCORE",
                    "seconds": round(dt, 1),
                    "run_dir": run_dir,
                    "config": cfg_path,
                    "overrides": overrides,
                    "wandb_project": per_algo_project,
                }
                summary.append(row)

                if score is not None:
                    cur = best_by_algo.get(algo_u)
                    if cur is None or float(score) > float(cur["score"]):
                        best_by_algo[algo_u] = row

    os.makedirs(sweep_root, exist_ok=True)

    csv_path = os.path.join(sweep_root, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        cols = ["algo", "preset", "seed", "episodes", "score", "status", "seconds", "run_dir", "config", "wandb_project"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary:
            w.writerow({k: r.get(k) for k in cols})

    best_path = os.path.join(sweep_root, "best_per_algo.yaml")
    out = {}
    for algo_u, r in best_by_algo.items():
        out[algo_u] = {
            "score": r["score"],
            "preset": r["preset"],
            "overrides": r["overrides"],
            "seed": r["seed"],
            "episodes": r["episodes"],
            "run_dir": r["run_dir"],
            "config": r["config"],
            "wandb_project": r["wandb_project"],
        }
    with open(best_path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("\n==================== BEST PER ALGORITHM ====================")
    if not best_by_algo:
        print("No successful runs produced a score (check eval settings / manifest writing).")
    else:
        for algo_u, r in best_by_algo.items():
            print(f"{algo_u}: best_score={r['score']} preset={r['preset']} seed={r['seed']}")
            print(f"      wandb_project={r['wandb_project']}")
            print(f"      overrides={r['overrides']}")
            print(f"      run_dir={r['run_dir']}")
            print(f"      config={r['config']}")
    print("============================================================\n")

    print(f"[DONE] Summary: {csv_path}")
    print(f"[DONE] Best per algo: {best_path}")


if __name__ == "__main__":
    main()
