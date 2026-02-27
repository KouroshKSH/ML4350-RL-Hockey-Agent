"""
A Round Robin Tournament Orchestrator for our own agents, basically an internal tournament.
"""

import os
import sys
import re
import glob
import yaml
import torch
import pickle
import numpy as np
import datetime
import argparse
import wandb
import hashlib
import math
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Any, Tuple
from multiprocessing import Pool
from itertools import combinations
from openskill.models import PlackettLuce
import csv

# Project path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Project root:\n{PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)

import hockey.hockey_env_original as h_env
from agent.external_models import actor_spec_from_config, load_obs_norm_npz
from agent.normalization import RunningMeanStd  # type compatibility

# requested default values (ignores other algos since we didn't need them)
DEFAULT_ALGOS = ["TD3", "TQC", "SAC"]
DEFAULT_SCORE_THRESHOLD = 0.88
GAMES_PER_MATCH = 20

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CONFIG_PATH = os.path.join(THIS_DIR, "tournament_config.yaml")

TRAJ_DIR = os.environ.get(
    "HOCKEY_TRAJ_DIR",
    os.path.join(PROJECT_ROOT, "internal_leaderboard", "trajectories"),
)

LEADERBOARD_DIR = os.environ.get(
    "HOCKEY_LEADERBOARD_DIR",
    os.path.join(PROJECT_ROOT, "internal_leaderboard", "leaderboards"),
)

# cache per process for every agent (worker-local)
_AGENT_CACHE: Dict[str, "LoadedAgent"] = {}


def _sanitize_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s.strip("_")[:200]


def _resolve_results_root(results_root_raw: Optional[str]) -> str:
    """
    - absolute paths are used as they are
    - relative paths are assigned relative to PROJECT_ROOT
    """
    results_root_raw = str(results_root_raw or "results")
    results_root_raw = os.path.expanduser(results_root_raw)
    if os.path.isabs(results_root_raw):
        return os.path.abspath(results_root_raw)
    return os.path.abspath(os.path.join(PROJECT_ROOT, results_root_raw))


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _stable_seed(*parts: Any) -> int:
    """
    stable per-game seed (0 .. 2**31-1) derived from arbitrary identifiers.
    to keep tournaments reproducible and to avoid correlated RNG across processes.
    """
    s = "|".join(map(str, parts)).encode("utf-8", errors="ignore")
    h = hashlib.blake2b(s, digest_size=8).digest()
    return int.from_bytes(h, "little") & 0x7FFFFFFF


def score_rate(w: int, l: int, t: int) -> float:
    n = w + l + t
    return (w + 0.5 * t) / n if n > 0 else 0.5


def wilson_lower_bound(p: float, n: int, z: float = 1.96) -> float:
    """
    Wilson lower bound for a proportion.
    as a conservative estimate of per-opponent "score_rate" with finite samples
    """
    if n <= 0:
        return 0.0
    denom = 1.0 + (z * z) / n
    center = p + (z * z) / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)
    return (center - margin) / denom


def discover_internal_models(
    results_root: str,
    env_name: str,
    algos_allow: List[str],
    score_threshold: float,
    include_segments: Optional[List[str]] = None,
    max_models: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    find the saved model entries from:
      results/<ALGO>/<env_name>/<RUN_NAME>/best_models_manifest.yaml

    return the descriptors with absolute paths:
      {
        "algo", "run_dir", "run_name", "segment", "best_score",
        "actor_path", "norm_path", "config_path", "display_name"
      }
    """
    allow = set([str(a).upper() for a in (algos_allow or [])])
    include_segments = include_segments or ["best", "mid", "late"]
    include = set([str(s).lower() for s in include_segments])
    threshold = float(score_threshold)

    project_root_for_manifest = os.path.dirname(os.path.abspath(results_root))

    def _resolve_manifest_path(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        p = os.path.expanduser(str(p))
        if os.path.isabs(p):
            return os.path.abspath(p)
        return os.path.abspath(os.path.join(project_root_for_manifest, p))

    manifest_glob = os.path.join(
        results_root, 
        "*", env_name, 
        "*", "best_models_manifest.yaml"
    )
    manifests = sorted(
        [
            p for p in glob.glob(manifest_glob) if os.path.isfile(p)
        ]
    )

    entries_out: List[Dict[str, Any]] = []
    seen_actor_paths: set = set()

    for man_path in manifests:
        run_dir = os.path.dirname(os.path.abspath(man_path))
        cfg_path = os.path.abspath(os.path.join(run_dir, "config.yaml"))
        if not os.path.exists(cfg_path):
            continue

        man = _load_yaml(man_path)
        algo = str(man.get("algo", "")).upper()
        if allow and algo not in allow:
            continue

        for e in (man.get("entries", []) or []):
            seg = str(e.get("segment", "")).lower()
            if seg not in include:
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

            if norm_path and (not os.path.exists(norm_path)):
                norm_path = None

            run_name = os.path.basename(run_dir.rstrip(os.sep))

            # make display_name UNIQUE by appending a short hash of the actor checkpoint path.
            actor_hash = hashlib.blake2b(os.path.abspath(actor_path).encode("utf-8"), digest_size=4).hexdigest()
            disp = f"{algo}__{run_name}__{seg}__score{best_score:.3f}__{actor_hash}"

            entries_out.append({
                "algo": algo,
                "run_dir": run_dir,
                "run_name": run_name,
                "segment": seg,
                "best_score": best_score,
                "actor_path": os.path.abspath(actor_path),
                "norm_path": os.path.abspath(norm_path) if norm_path else None,
                "config_path": os.path.abspath(cfg_path),
                "display_name": disp,
            })
            seen_actor_paths.add(actor_path)

            if max_models is not None and len(entries_out) >= int(max_models):
                return entries_out

    return entries_out


class LoadedAgent:
    """
    load an actor checkpoint + optional obs normalizer.
    """

    def __init__(self, descriptor: dict, state_dim: int = 18, default_action_dim: int = 4):
        self.desc = dict(descriptor)
        self.state_dim = int(state_dim)

        # make it defensive (important for workers)
        for k in ("actor_path", "norm_path", "config_path"):
            if self.desc.get(k):
                p = os.path.expanduser(str(self.desc[k]))
                if not os.path.isabs(p):
                    p = os.path.join(PROJECT_ROOT, p)
                self.desc[k] = os.path.abspath(p)

        actor_path = self.desc.get("actor_path")
        cfg_path = self.desc.get("config_path")

        if not actor_path or not os.path.exists(actor_path):
            raise FileNotFoundError(f"missing actor checkpoint: {actor_path}")
        if not cfg_path or not os.path.exists(cfg_path):
            raise FileNotFoundError(f"missing run config: {cfg_path}")

        run_cfg = _load_yaml(cfg_path)
        self.action_dim = int(run_cfg.get("action_dim", default_action_dim))

        algo = str(self.desc["algo"]).upper()
        actor_cls, actor_kwargs, has_sample = actor_spec_from_config(
            algo=algo,
            run_cfg=run_cfg,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )
        self.has_sample = bool(has_sample)

        self.actor = actor_cls(**actor_kwargs).to("cpu")
        self.actor.eval()

        sd = torch.load(actor_path, map_location="cpu")
        if not isinstance(sd, dict):
            raise ValueError(f"actor checkpoint is not a state_dict dict: {actor_path}")
        self.actor.load_state_dict(sd, strict=True)

        self.normalizer: Optional[RunningMeanStd] = None
        npath = self.desc.get("norm_path")
        if npath and os.path.exists(npath):
            try:
                self.normalizer = load_obs_norm_npz(npath, state_dim=self.state_dim)
            except Exception:
                self.normalizer = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.normalizer is not None:
            obs = self.normalizer.normalize_np(obs)

        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            if self.has_sample:
                _, _, det = self.actor.sample(s)
                a = det
            else:
                a = self.actor(s)
            return a.squeeze(0).cpu().numpy()


def _get_cached_agent(desc: dict) -> LoadedAgent:
    """
    Per-process cache keyed by actor_path.
    """
    key = os.path.abspath(os.path.expanduser(str(desc["actor_path"])))
    ag = _AGENT_CACHE.get(key)
    if ag is None:
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        ag = LoadedAgent(desc)
        _AGENT_CACHE[key] = ag
    return ag


def _short_model_tag(desc: dict) -> str:
    return f"{desc.get('display_name', 'UNKNOWN')} ({desc.get('actor_path', '?')})"


def _env_reset(env: Any, one_starting: bool, seed: int):
    """
    a helper that prefers explicit one_starting + seed to avoid implicit toggle bias.
    """
    try:
        return env.reset(one_starting=bool(one_starting), seed=int(seed))
    except TypeError:
        try:
            return env.reset(one_starting=bool(one_starting))
        except TypeError:
            return env.reset()


def run_single_game(args) -> Tuple[str, int, int, str, str, Optional[list], Optional[str]]:
    d1, d2, p1_id, p2_id, game_idx, save_data, one_starting, seed = args

    env = None
    history = []
    try:
        env = h_env.HockeyEnv()
        a1 = _get_cached_agent(d1)
        a2 = _get_cached_agent(d2)

        obs1, _ = _env_reset(env, one_starting=bool(one_starting), seed=int(seed))
        obs2 = env.obs_agent_two()

        done = False
        while not done:
            try:
                act1 = a1.act(obs1)
            except Exception as ex:
                msg = f"act_error(p1): {_short_model_tag(d1)} vs {_short_model_tag(d2)} | {repr(ex)}"
                return ("err", 0, 1, p1_id, p2_id, None, msg)

            try:
                act2 = a2.act(obs2)
            except Exception as ex:
                msg = f"act_error(p2): {_short_model_tag(d1)} vs {_short_model_tag(d2)} | {repr(ex)}"
                return ("err", 0, 2, p1_id, p2_id, None, msg)

            if save_data:
                history.append({"obs1": obs1, "obs2": obs2, "act1": act1, "act2": act2})

            try:
                obs1, _, done, _, info = env.step(np.hstack([act1, act2]))
            except Exception as ex:
                msg = f"step_error: {_short_model_tag(d1)} vs {_short_model_tag(d2)} | {repr(ex)}"
                return ("err", 0, 0, p1_id, p2_id, None, msg)

            obs2 = env.obs_agent_two()

        winner_env = info.get("winner")
        winner_seat = 1 if winner_env == 1 else (2 if winner_env == -1 else 0)
        return ("ok", winner_seat, 0, p1_id, p2_id, history if (save_data and history) else None, None)

    except Exception as ex:
        msg = f"game_error: {_short_model_tag(d1)} vs {_short_model_tag(d2)} | {repr(ex)}"
        return ("err", 0, 0, p1_id, p2_id, None, msg)

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def preflight_filter_loadable(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    skipped = 0

    for m in models:
        try:
            _ = LoadedAgent(m)
            kept.append(m)
        except Exception as ex:
            skipped += 1
            print(f"[skip] could not load model: {m.get('display_name','?')} | {repr(ex)}")

    if skipped > 0:
        print(f"Skipped {skipped} unloadable model(s).")

    return kept


def main():
    parser = argparse.ArgumentParser(description="Run internal round-robin leaderboard tournament.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--algos", type=str, nargs="+", default=DEFAULT_ALGOS, choices=DEFAULT_ALGOS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--results-root", type=str, default=os.environ.get("HOCKEY_RESULTS_ROOT", "results"))
    parser.add_argument("--env-name", type=str, default="hockey")
    parser.add_argument("--include-segments", type=str, nargs="+", default=["best", "mid", "late"])
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for matchup/game seeding.")
    args = parser.parse_args()

    os.makedirs(TRAJ_DIR, exist_ok=True)

    cfg_path = os.path.abspath(args.config)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Could not find config file: {cfg_path}")

    config = _load_yaml(cfg_path)
    settings = config.get("settings", {}) or {}
    num_workers = int(settings.get("num_workers", 8))
    save_every_n_games = int(settings.get("save_every_n_games", 10))
    save_trajectories = bool(settings.get("save_trajectories", True))

    results_root = _resolve_results_root(args.results_root)

    models = discover_internal_models(
        results_root=results_root,
        env_name=str(args.env_name),
        algos_allow=args.algos,
        score_threshold=float(args.threshold),
        include_segments=list(args.include_segments),
        max_models=args.max_models,
    )

    if not models:
        raise RuntimeError(
            "No eligible models found.\n"
            f"  looked for: {os.path.join(results_root, '*', args.env_name, '*', 'best_models_manifest.yaml')}\n"
            f"  algos: {args.algos}\n"
            f"  threshold: {args.threshold}\n"
            f"  segments: {args.include_segments}\n"
        )

    print(f"Discovered {len(models)} model(s) above threshold. Preflighting loadability...")
    models = preflight_filter_loadable(models)

    if len(models) < 2:
        raise RuntimeError(f"Need at least 2 loadable models to run a tournament, got {len(models)}.")

    agent_ids = [m["display_name"] for m in models]
    id_to_desc = {m["display_name"]: m for m in models}

    # guarantee no duplicate IDs
    counts = Counter(agent_ids)
    dups = [k for k, v in counts.items() if v > 1]
    if dups:
        raise RuntimeError(
            "Duplicate agent IDs detected (display_name collision). "
            "IDs must be unique. Examples:\n  " + "\n  ".join(dups[:10])
        )

    print(f"Selected {len(models)} loadable model(s) (algos={args.algos}, threshold={args.threshold}).")

    if GAMES_PER_MATCH <= 0:
        raise ValueError("GAMES_PER_MATCH must be > 0")

    if GAMES_PER_MATCH % 2 != 0:
        print("[warn] GAMES_PER_MATCH is odd; seat-swapping will be off by 1 game.")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(
        project="hockey_internal_leaderboard",
        name=f"leaderboard_{timestamp}",
        config={
            "settings": {
                "games_per_match": GAMES_PER_MATCH,
                "num_workers": num_workers,
                "save_every_n_games": save_every_n_games,
                "save_trajectories": save_trajectories,
                "seed": int(args.seed),
            },
            "selection": {
                "algos": args.algos,
                "threshold": float(args.threshold),
                "include_segments": list(args.include_segments),
                "results_root": results_root,
                "env_name": str(args.env_name),
                "num_models": len(models),
            },
        },
    )

    run_id = getattr(getattr(wandb, "run", None), "id", None) or "no_wandb_id"
    os.makedirs(LEADERBOARD_DIR, exist_ok=True)
    csv_path = os.path.join(LEADERBOARD_DIR, f"leaderboard_{timestamp}_{run_id}.csv")

    skill_model = PlackettLuce()
    ratings = {aid: skill_model.rating(name=aid) for aid in agent_ids}

    # Global totals
    stats = {
        aid: {"wins": 0, "losses": 0, "ties": 0, "errors": 0, "void": 0}
        for aid in agent_ids
    }

    # compete head to head from each agent's perspective
    # head2head[a][b] = {"w","l","t"} counts for a vs b
    head2head = defaultdict(lambda: defaultdict(lambda: {"w": 0, "l": 0, "t": 0}))

    pairs = list(combinations(agent_ids, 2))
    total_matchups = len(pairs)
    total_games = total_matchups * GAMES_PER_MATCH
    print(f"Total matchups: {total_matchups} | Total games: {total_games}")

    worker_errors = 0

    actor_path_by_id = {m["display_name"]: m["actor_path"] for m in models}
    norm_path_by_id = {m["display_name"]: (m.get("norm_path") or "") for m in models}

    # reuse a single Pool for the whole tournament.
    with Pool(processes=num_workers) as pool:
        for matchup_idx, (a_id, b_id) in enumerate(pairs, start=1):
            da, db = id_to_desc[a_id], id_to_desc[b_id]
            print(f"Match {matchup_idx}/{total_matchups}: {a_id} vs {b_id}")

            half = GAMES_PER_MATCH // 2
            tasks = []
            for i in range(GAMES_PER_MATCH):
                # seat swap: 
                # first half A as P1, 
                # second half B as P1 (or off by 1 if odd)
                swap = (i >= half)
                p1_id, p2_id = (b_id, a_id) if swap else (a_id, b_id)
                d1, d2 = (db, da) if swap else (da, db)

                # balanced starts: alternate "player1 starts" across games
                one_starting = (i % 2 == 0)

                # use stable seeds for each game
                seed = _stable_seed(args.seed, matchup_idx, a_id, b_id, i, int(swap))

                save_data = save_trajectories and (i % save_every_n_games == 0)
                tasks.append((d1, d2, p1_id, p2_id, i, save_data, one_starting, seed))

            results = pool.map(run_single_game, tasks)

            for i, (status, winner_seat, crashed_seat, p1_id, p2_id, history, err_msg) in enumerate(results):
                if status == "err":
                    worker_errors += 1

                    if crashed_seat == 1:
                        # p1 crashed => p1 loss, p2 win
                        stats[p1_id]["errors"] += 1
                        stats[p1_id]["losses"] += 1
                        stats[p2_id]["wins"] += 1

                        head2head[p2_id][p1_id]["w"] += 1
                        head2head[p1_id][p2_id]["l"] += 1

                        new_r = skill_model.rate([[ratings[p2_id]], [ratings[p1_id]]])
                        ratings[p2_id], ratings[p1_id] = new_r[0][0], new_r[1][0]

                    elif crashed_seat == 2:
                        # p2 crashed => p2 loss, p1 win
                        stats[p2_id]["errors"] += 1
                        stats[p2_id]["losses"] += 1
                        stats[p1_id]["wins"] += 1

                        head2head[p1_id][p2_id]["w"] += 1
                        head2head[p2_id][p1_id]["l"] += 1

                        new_r = skill_model.rate([[ratings[p1_id]], [ratings[p2_id]]])
                        ratings[p1_id], ratings[p2_id] = new_r[0][0], new_r[1][0]

                    else:
                        # if the attribution is unknown, count as "void" for both, skip rating update.
                        stats[p1_id]["errors"] += 1
                        stats[p2_id]["errors"] += 1
                        stats[p1_id]["void"] += 1
                        stats[p2_id]["void"] += 1

                    if worker_errors <= 20:
                        print(f"[warn] {err_msg}")
                    elif worker_errors == 21:
                        print("[warn] further game errors will be suppressed.")
                    continue

                # normal result path
                if winner_seat == 1:
                    stats[p1_id]["wins"] += 1
                    stats[p2_id]["losses"] += 1

                    head2head[p1_id][p2_id]["w"] += 1
                    head2head[p2_id][p1_id]["l"] += 1

                    new_r = skill_model.rate([[ratings[p1_id]], [ratings[p2_id]]])
                    ratings[p1_id], ratings[p2_id] = new_r[0][0], new_r[1][0]

                elif winner_seat == 2:
                    stats[p2_id]["wins"] += 1
                    stats[p1_id]["losses"] += 1

                    head2head[p2_id][p1_id]["w"] += 1
                    head2head[p1_id][p2_id]["l"] += 1

                    new_r = skill_model.rate([[ratings[p2_id]], [ratings[p1_id]]])
                    ratings[p2_id], ratings[p1_id] = new_r[0][0], new_r[1][0]

                else:
                    stats[p1_id]["ties"] += 1
                    stats[p2_id]["ties"] += 1

                    head2head[p1_id][p2_id]["t"] += 1
                    head2head[p2_id][p1_id]["t"] += 1

                    new_r = skill_model.rate([[ratings[p1_id]], [ratings[p2_id]]], ranks=[1, 1])
                    ratings[p1_id], ratings[p2_id] = new_r[0][0], new_r[1][0]

                if history:
                    safe1 = _sanitize_filename(p1_id)
                    safe2 = _sanitize_filename(p2_id)
                    filename = f"{timestamp}_{safe1}_vs_{safe2}_g{i}.pkl"
                    folderpath = os.path.join(TRAJ_DIR, timestamp)
                    filepath = os.path.join(folderpath, filename)
                    os.makedirs(folderpath, exist_ok=True)
                    with open(filepath, "wb") as f:
                        pickle.dump(history, f)
                    print(f"  Saved trajectory: {filename}")

    if worker_errors > 0:
        print(f"Finished with {worker_errors} game error(s).")

    # diagnostics to check robustness
    worst5_str: Dict[str, str] = {}
    min_sr: Dict[str, float] = {}
    bottom5_avg_sr: Dict[str, float] = {}
    bottom5_avg_wilson: Dict[str, float] = {}
    hard_counter_count: Dict[str, int] = {}

    for a in agent_ids:
        rows = []
        for b in agent_ids:
            if b == a:
                continue
            w = head2head[a][b]["w"]
            l = head2head[a][b]["l"]
            t = head2head[a][b]["t"]
            n = w + l + t
            if n <= 0:
                continue
            sr = score_rate(w, l, t)
            wlb = wilson_lower_bound(sr, n, z=1.96)
            rows.append((sr, wlb, b, w, l, t, n))

        rows.sort(key=lambda x: x[0])  # worst first by score rate

        if rows:
            worst = rows[:5]
            worst5_str[a] = "; ".join([f"{b}: sr={sr:.2f} ({w}/{l}/{t})" for (sr, wlb, b, w, l, t, n) in worst])
            min_sr[a] = rows[0][0]
            k = min(5, len(rows))
            bottom5_avg_sr[a] = sum(r[0] for r in rows[:k]) / k
            bottom5_avg_wilson[a] = sum(r[1] for r in rows[:k]) / k
            hard_counter_count[a] = sum(1 for (sr, *_rest) in rows if sr < 0.25)
        else:
            worst5_str[a] = ""
            min_sr[a] = 0.5
            bottom5_avg_sr[a] = 0.5
            bottom5_avg_wilson[a] = 0.0
            hard_counter_count[a] = 0

    # everyone should have equal total games participated
    expected = (len(agent_ids) - 1) * GAMES_PER_MATCH
    mismatches = []
    for aid in agent_ids:
        s = stats[aid]
        played = s["wins"] + s["losses"] + s["ties"] + s["void"]
        if played != expected:
            mismatches.append((aid, played, expected, s["wins"], s["losses"], s["ties"], s["void"], s["errors"]))
    if mismatches:
        print("[warn] Unequal games-per-agent detected (including void). This should NOT happen.")
        print("[warn] Showing up to 10 examples:")
        for row in mismatches[:10]:
            print(" ", row)

    # the columns for the W&B leaderboard
    columns = [
        "Rank", "Agent",
        "Ordinal(mu-3sigma)", "Mu", "Sigma",
        "W/L/T", "Void", "Errors",
        "MinScoreRate", "Bottom5AvgScoreRate", "Bottom5AvgWilsonLB",
        "HardCounterCount", "Worst5Opponents", # added hard-counters in v2 of leaderboard
        "ActorPath", "NormPath",
    ]

    table = wandb.Table(columns=columns)

    sorted_ids = sorted(agent_ids, key=lambda x: ratings[x].ordinal(), reverse=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for i, aid in enumerate(sorted_ids):
            r = ratings[aid]
            s = stats[aid]

            row = [
                i + 1,
                aid,
                round(r.ordinal(), 3),
                round(r.mu, 3),
                round(r.sigma, 3),
                f"{s['wins']}/{s['losses']}/{s['ties']}",
                int(s["void"]),
                int(s["errors"]),
                round(min_sr.get(aid, 0.5), 4),
                round(bottom5_avg_sr.get(aid, 0.5), 4),
                round(bottom5_avg_wilson.get(aid, 0.0), 4),
                int(hard_counter_count.get(aid, 0)),
                worst5_str.get(aid, ""),
                actor_path_by_id.get(aid, ""),
                norm_path_by_id.get(aid, ""),
            ]

            table.add_data(*row)
            writer.writerow(row)

    print(f"[info] Saved local leaderboard CSV: {csv_path}")

    wandb.log({"Leaderboard": table})
    wandb.finish()


if __name__ == "__main__":
    main()
