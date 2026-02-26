import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
import argparse
import datetime
import wandb
import random
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hockey.hockey_env as h_env
from agent.ddpg_agent import DDPGAgent
from agent.td3_agent import TD3Agent
from agent.tqc_agent import TQCAgent
from agent.sac_agent import SACAgent
from agent.redq_agent import REDQAgent
from agent.droq_agent import DroQAgent
from agent.crossq_agent import CrossQAgent

from agent.memory import ReplayBuffer
from agent.normalization import RunningMeanStd
from agent.augment import flip_y_obs, flip_y_action
from agent.opponents import OpponentManager
from agent.league import PSROLeague


def _cfgget(cfg, key, default):
    v = cfg.get(key, default)
    return default if v is None else v


def _get_phase(cfg: dict, episode: int) -> dict:
    phases = cfg.get("phases", []) or []
    for ph in phases:
        try:
            until = int(ph.get("until_episode", -1))
        except Exception:
            continue
        if until >= 0 and episode < until:
            return dict(ph)
    return dict(phases[-1]) if phases else {}


def _set_env_mode(env: h_env.HockeyEnv, mode_name: str | None):
    """
    we set env.mode directly and call reset() without mode=... 
    as the provided HockeyEnv.reset(mode=...) had some issues.
    """
    name = str(mode_name or "NORMAL").upper()
    if name in ("NORMAL", "0"):
        env.mode = h_env.Mode.NORMAL
    elif name in ("TRAIN_SHOOTING", "SHOOTING", "1"):
        env.mode = h_env.Mode.TRAIN_SHOOTING
    elif name in ("TRAIN_DEFENSE", "DEFENSE", "2"):
        env.mode = h_env.Mode.TRAIN_DEFENSE
    else:
        env.mode = h_env.Mode.NORMAL


def _merge_reward_cfg(base_reward_cfg: dict, phase: dict) -> dict:
    rc = dict(base_reward_cfg or {})
    overrides = (phase.get("reward_overrides") or {})
    if overrides:
        rc.update(overrides)
        if "danger" in overrides and isinstance(overrides["danger"], dict):
            d = dict((base_reward_cfg or {}).get("danger", {}) or {})
            d.update(overrides["danger"])
            rc["danger"] = d
    return rc


def log_heatmap(episode, positions, algorithm):
    if not positions:
        return None
    pos_array = np.array(positions)

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.hist2d(
        pos_array[:, 0],
        pos_array[:, 1],
        bins=30,
        cmap='Blues',
        range=[[-15, 15], [-10, 10]],
    )
    ax.set_title(f"Puck Heatmap - Algorithm {algorithm} - Episode {episode}")
    fig.colorbar(h[3], ax=ax)

    wb_image = wandb.Image(fig)
    plt.close(fig) # to avoid wasting RAM
    return wb_image


def _save_obs_norm_npz(path, obs_normalizer):
    np.savez(
        path,
        mean=obs_normalizer.mean,
        var=obs_normalizer.var,
        count=obs_normalizer.count,
        clip=getattr(obs_normalizer, "clip", 5.0),
    )

def _load_obs_norm_npz(path, obs_normalizer):
    data = np.load(path)
    obs_normalizer.mean = data["mean"]
    obs_normalizer.var = data["var"]
    obs_normalizer.count = data["count"]
    if "clip" in data.files:
        obs_normalizer.clip = float(data["clip"])


def _has_self_play_pool(opp_mgr):
    return hasattr(opp_mgr, "pool") and (len(opp_mgr.pool) > 0)


def _normalize_for_opponent(obs_normalizer, kind, obs_agent2, ctx=None):
    if kind == "self_play" and obs_normalizer is not None:
        return np.asarray(obs_normalizer.normalize_np(obs_agent2), dtype=np.float32)

    if kind == "external" and ctx is not None:
        ext_norm = ctx.get("obs_norm", None)
        if ext_norm is not None:
            return np.asarray(ext_norm.normalize_np(obs_agent2), dtype=np.float32)

    return np.asarray(obs_agent2, dtype=np.float32)


def _has_external_pool(opp_mgr):
    return hasattr(opp_mgr, "external_pool") and (len(opp_mgr.external_pool) > 0)

def _sample_opponent_kind_with_phase(cfg, opp_mgr, episode: int, phase: dict) -> str:
    # read the opponent mix from phase, else global, else defaults
    mix = phase.get("opponent_mix")
    if not isinstance(mix, dict) or not mix:
        mix = cfg.get("opponent_mix")
    if not isinstance(mix, dict) or not mix:
        mix = {"weak": 0.35, "strong": 0.35, "self_play": 0.30, "external": 0.0}

    p_w   = float(mix.get("weak", 0.0))
    p_s   = float(mix.get("strong", 0.0))
    p_sp  = float(mix.get("self_play", 0.0))
    p_ext = float(mix.get("external", 0.0))

    # self-play availability + ramp
    sp_cfg = (cfg.get("self_play") or {})
    sp_enabled = bool(sp_cfg.get("enabled", True))
    snap_start = int(sp_cfg.get("snapshot_start_episode", getattr(opp_mgr, "snapshot_min_episode", 400)))

    cur_cfg = (cfg.get("curriculum") or {})
    cur_enabled = bool(cur_cfg.get("enabled", True))
    ramp_eps = int(cur_cfg.get("self_play_ramp_episodes", 1000))

    if (not sp_enabled) or (not _has_self_play_pool(opp_mgr)):
        p_sp = 0.0
    elif p_sp > 0.0:
        if episode < snap_start:
            p_sp = 0.0
        elif cur_enabled and ramp_eps > 0:
            ramp = min(1.0, max(0.0, (episode - snap_start) / float(ramp_eps)))
            p_sp *= ramp

    # external availability + ramp
    ext_cfg = (cfg.get("external_pool") or {})
    ext_enabled = bool(ext_cfg.get("enabled", False))
    if (not ext_enabled) or (not _has_external_pool(opp_mgr)):
        p_ext = 0.0
    elif p_ext > 0.0:
        ext_start = int(ext_cfg.get("start_episode", 0))
        ext_ramp = int(ext_cfg.get("ramp_episodes", 0))
        if episode < ext_start:
            p_ext = 0.0
        elif ext_ramp > 0:
            ramp = min(1.0, max(0.0, (episode - ext_start) / float(ext_ramp)))
            p_ext *= ramp

    probs = np.array(
        [
            max(p_w, 0.0), max(p_s, 0.0), max(p_sp, 0.0), max(p_ext, 0.0)
        ], 
        dtype=np.float64,
    )
    s = float(probs.sum())
    if s <= 0.0:
        return "strong"
    probs /= s

    return random.choices(["weak", "strong", "self_play", "external"], weights=probs.tolist(), k=1)[0]

@torch.no_grad()
def evaluate(agent, cfg, opp_mgr, obs_normalizer=None):
    eval_cfg = cfg.get("eval", {}) or {}
    if not eval_cfg.get("enabled", True):
        return None

    n_eps = int(eval_cfg.get("episodes_per_opponent", 50))
    step_count = int(cfg["step_count"])

    # keep eval fixed (False) to act as a consistent testing for the agent
    env = h_env.HockeyEnv(random_start=False)

    opponents = eval_cfg.get("opponents", ["weak", "strong"])
    opponents = list(dict.fromkeys(opponents))

    if "external" in opponents and not _has_external_pool(opp_mgr):
        raise RuntimeError(
            "Eval requested 'external' opponent but external_pool is empty. "
            "This should have been caught by the preflight check."
        )
    if "self_play" in opponents and not _has_self_play_pool(opp_mgr):
        opponents = [k for k in opponents if k != "self_play"]


    results = {}
    total_score = 0.0
    total_games = 0

    for kind in opponents:
        wins = losses = ties = 0

        for _ in range(n_eps):
            obs, _ = env.reset()
            obs2 = env.obs_agent_two()
            info = {}

            # NOTE: snapshot fixed per episode
            if kind in ("self_play", "external"):
                kind_ep, ctx = opp_mgr.start_episode(kind)
            else:
                kind_ep, ctx = kind, None

            for _t in range(step_count):
                obs_in = obs_normalizer.normalize_np(obs) if obs_normalizer is not None else obs
                a1 = agent.select_action(np.array(obs_in), noise=0.0)

                obs2_in = _normalize_for_opponent(obs_normalizer, kind_ep, obs2, ctx=ctx)
                a2 = opp_mgr.act(kind_ep, obs2_in, ctx=ctx)

                obs, _r, done, _tr, info = env.step(np.hstack([a1, a2]))
                obs2 = env.obs_agent_two()
                if done:
                    break

            w = int(info.get("winner", 0))
            if w == 1:
                wins += 1
            elif w == 0:
                ties += 1
            else:
                losses += 1

        score = (wins + 0.5 * ties) / max(1, (wins + ties + losses))
        results[kind] = {"wins": wins, "ties": ties, "losses": losses, "score": score}

        total_score += score * (wins + ties + losses)
        total_games += (wins + ties + losses)

    results["total_score"] = total_score / max(1, total_games)
    return results

def termlog(msg: str):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def train():
    parser = argparse.ArgumentParser()
    # path to config file for training
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/tqc_private.yaml"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--user", type=str, default=None)
    args = parser.parse_args()

    # using yaml config file
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # set seeds for reproducibility
    if args.seed is not None:
        cfg['seed'] = int(args.seed)
    if args.user is not None:
        cfg['user'] = str(args.user)

    np.random.seed(int(cfg['seed']))
    torch.manual_seed(int(cfg['seed']))
    random.seed(int(cfg['seed']))

    algo = str(cfg.get("algo", "TQC")).upper()
    env_name = str(cfg.get("env", "hockey"))

    # account for random starting condition for player 1
    rand_start = cfg.get("rand_start_cond_player1", False)
    env = h_env.HockeyEnv(random_start=rand_start)
    state_dim = env.observation_space.shape[0]
    action_dim = int(cfg.get("action_dim", 4))

    if algo == "DDPG":
        expl = cfg.get("exploration", {}) or {}
        agent = DDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=float(cfg["lr"]),
            gamma=float(cfg["gamma"]),
            tau=float(cfg["tau"]),
            exploration_noise_type=str(expl.get("type", "gaussian")),
            pink_rows=int(expl.get("pink_rows", 16)),
            seed=cfg.get("seed", None),
        )

    elif algo == "TD3":
        td3_cfg = cfg.get("td3", {}) or {}
        policy_noise = float(td3_cfg.get("policy_noise", cfg.get("policy_noise", 0.2)))
        noise_clip   = float(td3_cfg.get("noise_clip",   cfg.get("noise_clip", 0.5)))
        policy_delay = int(td3_cfg.get("policy_delay",   cfg.get("policy_delay", 2)))

        expl = cfg.get("exploration", {}) or {}

        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=float(td3_cfg.get("lr", cfg["lr"])),
            gamma=float(cfg["gamma"]),
            tau=float(cfg["tau"]),
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_delay=policy_delay,
            exploration_noise_type=str(expl.get("type", "gaussian")),
            pink_rows=int(expl.get("pink_rows", 16)),
            seed=cfg.get("seed", None),
        )

    elif algo == "TQC":
        tqc_cfg = cfg.get("tqc", {}) or {}
        agent = TQCAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_lr=float(tqc_cfg.get("actor_lr", cfg["lr"])),
            critic_lr=float(tqc_cfg.get("critic_lr", cfg["lr"])),
            gamma=float(cfg["gamma"]),
            tau=float(cfg["tau"]),
            n_critics=int(tqc_cfg.get("n_critics", 5)),
            n_quantiles=int(tqc_cfg.get("n_quantiles", 25)),
            top_quantiles_to_drop=int(tqc_cfg.get("top_quantiles_to_drop", 2)),
            target_entropy=tqc_cfg.get("target_entropy", None),
            alpha=tqc_cfg.get("alpha", "auto"),
            init_alpha=float(tqc_cfg.get("init_alpha", 0.2)),
        )
    elif algo == "SAC":
        sac_cfg = cfg.get("sac", {}) or {}
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_lr=float(sac_cfg.get("actor_lr", cfg["lr"])),
            critic_lr=float(sac_cfg.get("critic_lr", cfg["lr"])),
            gamma=float(cfg["gamma"]),
            tau=float(cfg["tau"]),
            alpha=sac_cfg.get("alpha", "auto"),
            init_alpha=float(sac_cfg.get("init_alpha", 0.2)),
            target_entropy=sac_cfg.get("target_entropy", None),
            actor_hidden_dim=int(sac_cfg.get("actor_hidden_dim", 256)),
            critic_hidden_dim=int(sac_cfg.get("critic_hidden_dim", 256)),
        )

    elif algo == "REDQ":
        redq_cfg = cfg.get("redq", {}) or {}
        agent = REDQAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            n_critics=int(redq_cfg.get("n_critics", 10)),
            target_subset=int(redq_cfg.get("target_subset", 2)),
            actor_lr=float(redq_cfg.get("actor_lr", cfg["lr"])),
            critic_lr=float(redq_cfg.get("critic_lr", cfg["lr"])),
            gamma=float(cfg["gamma"]),
            tau=float(cfg["tau"]),
            alpha=redq_cfg.get("alpha", "auto"),
            init_alpha=float(redq_cfg.get("init_alpha", 0.2)),
            target_entropy=redq_cfg.get("target_entropy", None),
            actor_hidden_dim=int(redq_cfg.get("actor_hidden_dim", 256)),
            critic_hidden_dim=int(redq_cfg.get("critic_hidden_dim", 256)),
        )

    elif algo == "DROQ":
        droq_cfg = cfg.get("droq", {}) or {}
        agent = DroQAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            n_critics=int(droq_cfg.get("n_critics", 2)),
            target_subset=int(droq_cfg.get("target_subset", 2)),
            dropout_p=float(droq_cfg.get("dropout_p", 0.01)),
            actor_lr=float(droq_cfg.get("actor_lr", cfg["lr"])),
            critic_lr=float(droq_cfg.get("critic_lr", cfg["lr"])),
            gamma=float(cfg["gamma"]),
            tau=float(cfg["tau"]),
            alpha=droq_cfg.get("alpha", "auto"),
            init_alpha=float(droq_cfg.get("init_alpha", 0.2)),
            target_entropy=droq_cfg.get("target_entropy", None),
            actor_hidden_dim=int(droq_cfg.get("actor_hidden_dim", 256)),
            critic_hidden_dim=int(droq_cfg.get("critic_hidden_dim", 256)),
        )

    elif algo == "CROSSQ":
        crossq_cfg = cfg.get("crossq", {}) or {}
        agent = CrossQAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_lr=float(crossq_cfg.get("actor_lr", cfg["lr"])),
            critic_lr=float(crossq_cfg.get("critic_lr", cfg["lr"])),
            gamma=float(cfg["gamma"]),
            alpha=crossq_cfg.get("alpha", "auto"),
            init_alpha=float(crossq_cfg.get("init_alpha", 0.2)),
            target_entropy=crossq_cfg.get("target_entropy", None),
            actor_hidden_dim=int(crossq_cfg.get("actor_hidden_dim", 256)),
            critic_hidden_dim=int(crossq_cfg.get("critic_hidden_dim", 1024)),
            brn_momentum=float(crossq_cfg.get("brn_momentum", 0.01)),
            brn_rmax=float(crossq_cfg.get("brn_rmax", 3.0)),
            brn_dmax=float(crossq_cfg.get("brn_dmax", 5.0)),
            brn_warmup_updates=int(crossq_cfg.get("brn_warmup_updates", 20000)),
        )

    else:
        raise ValueError(
            f"Unknown algo '{algo}'. Expected one of: DDPG, TD3, TQC, SAC, REDQ, DROQ, CROSSQ."
        )

    # NOTE: make sure that the self-play snapshots can...
    # reconstruct the actor with the SAME architecture used for training
    if algo in ("SAC", "REDQ", "DROQ"):
        # uses GaussianPolicy(state_dim, action_dim, hidden_dim=...)
        sub = cfg.get(algo.lower(), {}) or {}
        agent.actor_init_kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": int(sub.get("actor_hidden_dim", 256)),
        }

    elif algo == "CROSSQ":
        crossq_cfg = cfg.get("crossq", {}) or {}
        agent.actor_init_kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": int(crossq_cfg.get("actor_hidden_dim", 256)),
            "brn_momentum": float(crossq_cfg.get("brn_momentum", 0.01)),
            "brn_rmax": float(crossq_cfg.get("brn_rmax", 3.0)),
            "brn_dmax": float(crossq_cfg.get("brn_dmax", 5.0)),
            "brn_warmup_updates": int(crossq_cfg.get("brn_warmup_updates", 20000)),
        }

    # obs normalization
    obs_norm_cfg = cfg.get("obs_norm", {}) or {}
    obs_normalizer = None
    if bool(obs_norm_cfg.get("enabled", False)):
        obs_normalizer = RunningMeanStd(shape=(state_dim,), clip=float(obs_norm_cfg.get("clip", 5.0)))

    resume_cfg = cfg.get("resume", {}) or {}
    resume_enabled = bool(resume_cfg.get("enabled", False))
    start_episode = int(resume_cfg.get("start_episode", 0)) if resume_enabled else 0

    if resume_enabled:
        actor_path = str(resume_cfg.get("actor_path", ""))
        if not actor_path or not os.path.exists(actor_path):
            raise FileNotFoundError(f"resume.actor_path not found: {actor_path}")

        sd = torch.load(actor_path, map_location=agent.device)
        agent.actor.load_state_dict(sd, strict=True)

        # TD3/DDPG: keep target actor consistent
        if hasattr(agent, "actor_target") and agent.actor_target is not None:
            agent.actor_target.load_state_dict(agent.actor.state_dict(), strict=True)

        # load the observataions if enabled
        norm_path = str(resume_cfg.get("obs_norm_path", ""))
        if obs_normalizer is not None:
            if not norm_path or not os.path.exists(norm_path):
                raise FileNotFoundError(f"resume.obs_norm_path not found: {norm_path}")
            _load_obs_norm_npz(norm_path, obs_normalizer)

        termlog(f"[RESUME] Loaded actor: {actor_path}")
        if obs_normalizer is not None:
            termlog(f"[RESUME] Loaded obs_norm: {norm_path}")
        termlog(f"[RESUME] Starting at episode={start_episode}")

    # result dirs: unique run name with timestamp
    # Format: YYYY-MM-DD_HH-MM-SS_seed_{number}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("results", algo, env_name, f"{timestamp}_{cfg.get('user','user')}_seed{cfg['seed']}")
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    cfg_dump_path = os.path.join(run_dir, "config.yaml")
    with open(cfg_dump_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # ID of each experiment
    algo_name = str(cfg.get('algo', 'ALG')).upper()
    run_id = f"{algo_name}_{timestamp}_{cfg.get('user','user')}_seed{cfg['seed']}"
                                                                     
    # wandb project
    wandb_project = cfg.get("wandb_project", "RL-hockey")
    wandb.init(
        project=wandb_project, 
        name=run_id, 
        config=cfg
    )

    opp_mgr = OpponentManager(
        cfg, 
        state_dim=state_dim, 
        action_dim=action_dim, 
        device=agent.device
    )

    if resume_enabled:
        try:
            opp_mgr.maybe_add_snapshot(start_episode, agent, strength=None)
        except TypeError:
            opp_mgr.maybe_add_snapshot(start_episode, agent)

    # external pool (fail fast before phases start)
    ext_cfg = (cfg.get("external_pool") or {})
    if bool(ext_cfg.get("enabled", False)):
        min_models = int(ext_cfg.get("min_models", 1))

        if not hasattr(opp_mgr, "external_pool"):
            raise RuntimeError(
                "external_pool.enabled=true but OpponentManager has no 'external_pool'. "
                "You likely haven't added external support in agent/opponents.py."
            )

        n_ext = len(opp_mgr.external_pool)
        if n_ext < min_models:
            raise RuntimeError(
                f"external_pool.enabled=true but loaded {n_ext} external models "
                f"(min_models={min_models}). Check external_pool.results_root / score_threshold / algos / include_segments."
            )

        termlog(f"[external_pool] loaded {n_ext} external opponents (min_models={min_models})")

    league = PSROLeague(cfg, device=agent.device)

    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, max_size=cfg['buffer_size'])
    if obs_normalizer is not None:
        replay_buffer.set_obs_normalizer(obs_normalizer)

    sym_cfg = cfg.get("symmetry", {}) or {}
    symmetry_enabled = bool(sym_cfg.get("enabled", False))
    symmetry_prob = float(sym_cfg.get("prob", 0.5))
    symmetry_mode = str(sym_cfg.get("mode", "flip_y")).lower()

    warmup_steps = int(resume_cfg.get("warmup_steps", cfg.get("warmup_steps", 5000))) if resume_enabled else int(cfg.get("warmup_steps", 5000))
    updates_per_step = int(cfg.get("updates_per_step", 1))
    
    # reward shaping section
    reward_cfg = cfg.get("reward", {}) or {}

    gamma_float = float(cfg.get("gamma", 0.99))

    # terminal reward (applied only on done steps; winner is in {-1,0,1})
    terminal_win_reward = float(reward_cfg.get("terminal_win_reward", 10.0))

    # proxy terms (bounded / scaled)
    closeness_weight = float(reward_cfg.get("closeness_weight", 0.25))
    touch_bonus = float(reward_cfg.get("touch_bonus", 1.0))
    opp_touch_weight = float(reward_cfg.get("opp_touch_weight", 0.75))

    # progress shaping (potential-based on puck x-position)
    use_potential_shaping = bool(reward_cfg.get("use_potential_shaping", True))
    progress_weight = float(reward_cfg.get("progress_weight", 2.0))

    # puck x-velocity proxy
    # keep it small if you use progress shaping
    puck_dir_w = float(reward_cfg.get("puck_direction_weight", 0.2))

    # gated shot bonus (prevents "shoot spam")
    shot_bonus = float(reward_cfg.get("shot_bonus", 0.3))
    shot_gate_x = float(reward_cfg.get("shot_gate_x", 0.0))
    shot_gate_vx = float(reward_cfg.get("shot_gate_vx", 0.0))
    shot_gate_abs_y = float(reward_cfg.get("shot_gate_abs_y", 1.5))

    # regularization
    action_l2_w = float(reward_cfg.get("action_l2_weight", 3e-4))

    danger_cfg = reward_cfg.get("danger", {}) or {}
    danger_enabled = bool(danger_cfg.get("enabled", False))
    danger_w = float(danger_cfg.get("weight", 0.05))
    danger_x = float(danger_cfg.get("x_threshold", -10.0))
    danger_vx = float(danger_cfg.get("vx_threshold", -0.1))

    eval_cfg = cfg.get("eval", {}) or {}
    eval_enabled = bool(eval_cfg.get("enabled", True))
    eval_interval = int(eval_cfg.get("interval_episodes", 50))

    # NOTE: Per-phase / per-window best checkpoint saving
    # Rules:
    #  - If phase name contains "phase1" or "phase2": save ONE best for that phase.
    #  - Otherwise: save TWO best models:
    #       * best from middle third of that phase
    #       * best from last third of that phase
    #
    # Each saved "best" consists of:
    #   - actor weights (.pth)
    #   - obs normalizer (.npz) if obs_norm is enabled
    total_eps = int(cfg.get("episodes", 0))
    seed = int(cfg.get("seed", 0))
    algo_tag = str(cfg.get("algo", algo)).upper()
    wb_uid = getattr(getattr(wandb, "run", None), "id", None) or run_id

    def _sanitize_name(s: str) -> str:
        s = str(s or "phase").strip()
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

    def _build_phase_meta(phases_cfg: list, total_eps: int):
        meta = []
        if not phases_cfg:
            return [{"idx": 0, "name": "phase_all", "start": 0, "end": total_eps, "cfg": {}}]

        prev = 0
        for i, ph in enumerate(phases_cfg):
            ph = dict(ph or {})
            name = str(ph.get("name", f"phase{i+1}"))
            start = int(prev)
            if start >= total_eps:
                break
            end = int(ph.get("until_episode", total_eps))
            end = min(end, total_eps)
            if end <= start:
                end = min(start + 1, total_eps)
            meta.append({"idx": i, "name": name, "start": start, "end": end, "cfg": ph})
            prev = end

        # append a trailing "tail" phase ONLY if yaml doesn't cover total_eps
        if prev < total_eps:
            meta.append({"idx": len(meta), "name": "phase_tail", "start": prev, "end": total_eps, "cfg": {}})
        return meta

    phase_meta = _build_phase_meta(cfg.get("phases", []) or [], total_eps)

    def _get_phase_meta_for_episode(ep: int) -> dict:
        for pm in phase_meta:
            if ep < int(pm["end"]):
                return pm
        return phase_meta[-1]

    def _restore_best_actor_from_phase(phase_idx: int) -> bool:
        """
        Restore actor weights from the best checkpoint of a phase.
        """
        for seg in ("late", "best", "mid"):
            tr = phase_best.get((phase_idx, seg), None)
            if not tr:
                continue
            path = tr.get("actor_path", None)
            if not path or not os.path.exists(path):
                continue

            sd = torch.load(path, map_location=agent.device)
            agent.actor.load_state_dict(sd, strict=True)

            # TD3/DDPG have actor_target: keep it in sync after a restore
            if hasattr(agent, "actor_target") and agent.actor_target is not None:
                agent.actor_target.load_state_dict(agent.actor.state_dict(), strict=True)

            # also restore obs norm if present
            if obs_normalizer is not None:
                npath = tr.get("norm_path", None)
                if npath and os.path.exists(npath):
                    _load_obs_norm_npz(npath, obs_normalizer)
            termlog(
                f"[PHASE-RESTORE] phase='{tr.get('phase_name')}' seg='{seg}' "
                f"ep={tr.get('best_episode')} score={tr.get('best_score')}"
            )
            return True
        return False


    phase_best = {}
    forced_eval_episodes = set()

    for pm in phase_meta:
        pn = str(pm["name"])
        pn_l = pn.lower()
        p_start, p_end = int(pm["start"]), int(pm["end"])
        length = max(1, p_end - p_start)
        a = p_start + (length // 3)        # start of middle third
        b = p_start + (2 * length // 3)    # start of last third

        # always force at least one eval at phase end.
        forced_eval_episodes.add(max(p_start, p_end - 1))

        if ("phase1" in pn_l) or ("phase2" in pn_l):
            phase_best[(pm["idx"], "best")] = {
                "phase_name": pn,
                "segment": "best",
                "start": p_start,
                "end": p_end,
                "best_score": -1e9,
                "best_episode": None,
                "actor_path": None,
                "norm_path": None,
            }
        else:
            # force an eval at the end of the middle third.
            forced_eval_episodes.add(max(p_start, min(b, p_end) - 1))

            phase_best[(pm["idx"], "mid")] = {
                "phase_name": pn,
                "segment": "mid",
                "start": a,
                "end": b,
                "best_score": -1e9,
                "best_episode": None,
                "actor_path": None,
                "norm_path": None,
            }
            phase_best[(pm["idx"], "late")] = {
                "phase_name": pn,
                "segment": "late",
                "start": b,
                "end": p_end,
                "best_score": -1e9,
                "best_episode": None,
                "actor_path": None,
                "norm_path": None,
            }

    best_manifest_path = os.path.join(run_dir, "best_models_manifest.yaml")

    def _write_best_manifest():
        out = {
            "algo": algo_tag,
            "wandb_id": wb_uid,
            "seed": seed,
            "total_episodes": total_eps,
            "entries": [],
        }
        for (pidx, seg), tr in sorted(phase_best.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            out["entries"].append({
                "phase_idx": int(pidx),
                "phase_name": tr["phase_name"],
                "segment": tr["segment"],
                "range": [int(tr["start"]), int(tr["end"])],
                "best_episode": tr["best_episode"],
                "best_score": tr["best_score"] if tr["best_episode"] is not None else None,
                "actor_path": tr["actor_path"],
                "norm_path": tr["norm_path"],
            })
        with open(best_manifest_path, "w") as f:
            yaml.safe_dump(out, f, sort_keys=False)

    def _maybe_save_best(episode: int, score: float, phase_idx: int, obs_normalizer):
        any_saved = False
        for seg in ("best", "mid", "late"):
            key = (phase_idx, seg)
            if key not in phase_best:
                continue
            tr = phase_best[key]

            # only consider saves inside that segment's episode window.
            if not (episode >= int(tr["start"]) and episode < int(tr["end"])):
                continue

            if float(score) > float(tr["best_score"]):
                # remove previous files so only one best exists per required slot.
                if tr.get("actor_path") and os.path.exists(tr["actor_path"]):
                    try:
                        os.remove(tr["actor_path"])
                    except Exception:
                        pass
                if tr.get("norm_path") and tr["norm_path"] and os.path.exists(tr["norm_path"]):
                    try:
                        os.remove(tr["norm_path"])
                    except Exception:
                        pass

                safe_phase = _sanitize_name(tr["phase_name"])
                prefix = f"{algo_tag}__{wb_uid}__{safe_phase}__tot{total_eps}__seed{seed}__{seg}__ep{episode}"
                actor_path = os.path.join(checkpoint_dir, f"{prefix}.pth")
                norm_path = os.path.join(checkpoint_dir, f"{prefix}.npz")

                torch.save(agent.actor.state_dict(), actor_path)
                wandb.save(actor_path)

                if obs_normalizer is not None:
                    _save_obs_norm_npz(norm_path, obs_normalizer)
                    wandb.save(norm_path)
                else:
                    norm_path = None

                tr["best_score"] = float(score)
                tr["best_episode"] = int(episode)
                tr["actor_path"] = actor_path
                tr["norm_path"] = norm_path

                termlog(
                    f"[BEST-SAVE] phase='{tr['phase_name']}' segment='{seg}' "
                    f"episode={episode} score={float(score):.3f} -> {os.path.basename(actor_path)}"
                )
                any_saved = True

        if any_saved:
            _write_best_manifest()
        return any_saved

    # NOTE: there are some legacy names (some scripts default to these). We'll populate them at the end.
    legacy_best_actor = os.path.join(checkpoint_dir, "actor_best.pth")
    legacy_best_norm = os.path.join(checkpoint_dir, "obs_norm_best.npz")

    last_eval_score = None

    win_history = []
    window_size = int(cfg['window_size'])
    total_env_steps = 0

    pm0 = _get_phase_meta_for_episode(start_episode)
    last_phase_idx_seen = int(pm0["idx"])

    for episode in range(start_episode, int(cfg["episodes"])):
        logs = {}
        pm = _get_phase_meta_for_episode(episode)
        current_phase_idx = int(pm["idx"])

        if current_phase_idx != last_phase_idx_seen:
            # restore from the phase that just ended
            _restore_best_actor_from_phase(last_phase_idx_seen)
            last_phase_idx_seen = current_phase_idx

        phase = dict(pm.get("cfg") or {})
        phase.setdefault("name", pm.get("name", ""))
        _set_env_mode(env, phase.get("env_mode", "NORMAL"))

        obs, _ = env.reset()
        expl = cfg.get("exploration", {}) or {}
        if bool(expl.get("reset_each_episode", True)) and hasattr(agent, "reset_exploration"):
            agent.reset_exploration()
        obs_agent2 = env.obs_agent_two()
        total_reward = 0.0
        puck_positions = []

        # phase-specific step budget
        phase_step_count = int(phase.get("step_count", int(cfg["step_count"])))

        # phase-specific opponent mix + self-play sampling strategy
        ph_mix = phase.get("opponent_mix", None)
        if isinstance(ph_mix, dict) and ph_mix:
            opp_mgr.p_weak = float(ph_mix.get("weak", opp_mgr.p_weak))
            opp_mgr.p_strong = float(ph_mix.get("strong", opp_mgr.p_strong))
            opp_mgr.p_self = float(ph_mix.get("self_play", opp_mgr.p_self))
            if hasattr(opp_mgr, "p_external"):
                opp_mgr.p_external = float(ph_mix.get("external", getattr(opp_mgr, "p_external", 0.0)))

        if "self_play_sampling" in phase:
            opp_mgr.set_self_play_sampling(str(phase.get("self_play_sampling")))

        # phase reward overrides
        reward_cfg_ep = _merge_reward_cfg(reward_cfg, phase)
        terminal_win_reward_ep = float(reward_cfg_ep.get("terminal_win_reward", terminal_win_reward))
        closeness_w_ep = float(reward_cfg_ep.get("closeness_weight", closeness_weight))
        touch_bonus_ep = float(reward_cfg_ep.get("touch_bonus", touch_bonus))
        opp_touch_w_ep = float(reward_cfg_ep.get("opp_touch_weight", opp_touch_weight))

        use_potential_shaping_ep = bool(reward_cfg_ep.get("use_potential_shaping", use_potential_shaping))
        progress_w_ep = float(reward_cfg_ep.get("progress_weight", progress_weight))

        puck_dir_w_ep = float(reward_cfg_ep.get("puck_direction_weight", puck_dir_w))

        shot_bonus_ep = float(reward_cfg_ep.get("shot_bonus", shot_bonus))
        shot_gate_x_ep = float(reward_cfg_ep.get("shot_gate_x", shot_gate_x))
        shot_gate_vx_ep = float(reward_cfg_ep.get("shot_gate_vx", shot_gate_vx))
        shot_gate_abs_y_ep = float(reward_cfg_ep.get("shot_gate_abs_y", shot_gate_abs_y))

        action_l2_w_ep = float(reward_cfg_ep.get("action_l2_weight", action_l2_w))

        danger_cfg_ep = reward_cfg_ep.get("danger", {}) or danger_cfg
        danger_enabled_ep = bool(danger_cfg_ep.get("enabled", danger_enabled))
        danger_w_ep = float(danger_cfg_ep.get("weight", danger_w))
        danger_x_ep = float(danger_cfg_ep.get("x_threshold", danger_x))
        danger_vx_ep = float(danger_cfg_ep.get("vx_threshold", danger_vx))


        opp_kind = _sample_opponent_kind_with_phase(cfg, opp_mgr, episode, phase)
        opp_kind, opp_ctx = opp_mgr.start_episode(opp_kind)

        for t in range(int(phase_step_count)):
            puck_positions.append([obs[12], obs[13]])

            if obs_normalizer is not None and obs_norm_cfg.get("update", True):
                obs_normalizer.update(obs)

            obs_in = obs_normalizer.normalize_np(obs) if obs_normalizer is not None else obs
            noise_ep = float(phase.get("noise", cfg.get("noise", 0.2)))
            a1 = agent.select_action(state=np.array(obs_in), noise=noise_ep)

            obs2_in = _normalize_for_opponent(obs_normalizer, opp_kind, obs_agent2, ctx=opp_ctx)
            a2 = opp_mgr.act(opp_kind, obs2_in, ctx=opp_ctx)

            next_obs, reward, done, _, info = env.step(np.hstack([a1, a2]))
            total_env_steps += 1

            # shaped reward (bounded, self-play friendly)
            info2 = env.get_info_agent_two()

            # terminal component (winner is 0 until done)
            shaped_reward = terminal_win_reward_ep * float(info.get("winner", 0.0))

            # scaled defensive urgency proxy (negative in own-half danger situations)
            shaped_reward += closeness_w_ep * float(info.get("reward_closeness_to_puck", 0.0))

            # possession dynamics (gain puck, deny opponent)
            shaped_reward += touch_bonus_ep * float(info.get("reward_touch_puck", 0.0))
            shaped_reward -= opp_touch_w_ep * float(info2.get("reward_touch_puck", 0.0))

            # potential-based progress shaping on puck x-position (policy-invariant shaping form)
            if use_potential_shaping_ep:
                phi = float(obs[12]) / 5.0
                phi_next = float(next_obs[12]) / 5.0
                shaped_reward += progress_w_ep * (gamma_float * phi_next - phi)

            # velocity proxy (keep small if using progress shaping)
            shaped_reward += puck_dir_w_ep * float(info.get("reward_puck_direction", 0.0))

            # gated shot bonus to learn "good" shots without shoot-spam
            if len(a1) > 3 and float(a1[3]) > 0.5:
                puck_x_n = float(next_obs[12])
                puck_y_n = float(next_obs[13])
                puck_vx_n = float(next_obs[14])
                if (puck_x_n > shot_gate_x_ep) and (puck_vx_n > shot_gate_vx_ep) and (abs(puck_y_n) < shot_gate_abs_y_ep):
                    shaped_reward += shot_bonus_ep

            # action regularization
            shaped_reward -= action_l2_w_ep * float(np.sum(np.square(a1)))

            # extra safety penalty (keep disabled if the above is used)
            if danger_enabled_ep:
                if obs[12] < danger_x_ep and obs[14] < danger_vx_ep:
                    shaped_reward -= danger_w_ep


            done_mask = 1.0 if done else 0.0

            if symmetry_enabled and random.random() < symmetry_prob:
                if symmetry_mode == "flip_y":
                    obs_aug = flip_y_obs(obs)
                    next_obs_aug = flip_y_obs(next_obs)
                    a1_aug = flip_y_action(a1)

                    replay_buffer.add(obs_aug, a1_aug, next_obs_aug, shaped_reward, done_mask)
                else:
                    replay_buffer.add(obs, a1, next_obs, shaped_reward, done_mask)
            else:
                replay_buffer.add(obs, a1, next_obs, shaped_reward, done_mask)

            obs = next_obs
            obs_agent2 = env.obs_agent_two()
            total_reward += shaped_reward

            if total_env_steps >= warmup_steps:
                train_metrics = None
                for _ in range(int(updates_per_step)):
                    train_metrics = agent.train(replay_buffer, batch_size=int(cfg["batch_size"]))

                # logging Critic Loss, Actor Loss, Alpha
                if isinstance(train_metrics, dict) and train_metrics:
                    logs.update({f"Train/{k}": v for k, v in train_metrics.items()})

            if done:
                break

        winner = int(info.get("winner", 0))
        opp_mgr.end_episode(opp_kind, opp_ctx, winner)

        win_history.append(1 if winner == 1 else 0)
        if len(win_history) > window_size:
            win_history.pop(0)
        success_rate = float(np.mean(win_history)) if win_history else 0.0

        logs.update({
            "Episode": episode,
            "TotalReward": total_reward,
            "Winner": winner,
            "Stats/Success_Rate": success_rate,
            "Stats/EnvSteps": total_env_steps,
            "Phase/name": str(phase.get("name", "")),
            "Phase/env_mode": str(phase.get("env_mode", "")),
            "Phase/step_count": int(phase_step_count),
            "Opponent/kind": str(opp_kind),
        })
       
        print_every = int(cfg.get("print_every_episodes", 1))
        if episode % print_every == 0:
            phase_name = str(phase.get("name", ""))
            msg = (
                f"Seed {cfg['seed']} | Algorithm {algo} | Steps {total_env_steps} | "
                f"Reward {total_reward:.2f} | Winner: {winner} | Opp: {opp_kind}"
            )
            if phase_name:
                msg += f" | Phase: {phase_name}"
            termlog(msg)


        # eval + best checkpoint selection
        if eval_enabled and episode > 0 and (episode % eval_interval == 0 or episode in forced_eval_episodes):
            eval_res = evaluate(agent, cfg, opp_mgr, obs_normalizer=obs_normalizer)
            if eval_res is not None:
                logs["Eval/total_score"] = float(eval_res.get("total_score", 0.0))
                for k, v in eval_res.items():
                    if isinstance(v, dict):
                        logs[f"Eval/{k}_score"] = float(v.get("score", 0.0))
                        logs[f"Eval/{k}_wins"] = int(v.get("wins", 0))
                        logs[f"Eval/{k}_ties"] = int(v.get("ties", 0))
                        logs[f"Eval/{k}_losses"] = int(v.get("losses", 0))

                last_eval_score = float(eval_res.get("total_score", 0.0))

                # Save best checkpoints according to per-phase rules
                _maybe_save_best(
                    episode=episode,
                    score=last_eval_score,
                    phase_idx=current_phase_idx,
                    obs_normalizer=obs_normalizer,
                )

        # heatmap logging
        if episode % 200 == 0 and puck_positions:
            heatmap_img = log_heatmap(episode, puck_positions, algo)
            if heatmap_img is not None:
                logs["Heatmap"] = heatmap_img

        strength_est = last_eval_score
        if strength_est is None:
            strength_est = logs.get("Stats/Success_Rate", None)

        try:
            opp_mgr.maybe_add_snapshot(episode, agent, strength=strength_est)
        except TypeError:
            opp_mgr.maybe_add_snapshot(episode, agent)

        # PSRO meta update. Produces a distribution over snapshot_ids.
        meta = league.maybe_update(episode, opp_mgr.pool, obs_normalizer=obs_normalizer)
        if meta is not None:
            opp_mgr.set_meta_probs(meta)
            logs["PSRO/updated"] = 1
            logs["PSRO/league_size"] = len(meta)
        else:
            logs["PSRO/updated"] = 0

        wandb.log(logs, step=total_env_steps)

        ckpt_every = int(cfg.get("checkpoint_interval_episodes", 200))
        if ckpt_every > 0 and episode > 0 and episode % ckpt_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"actor_ep{episode}.pth")
            torch.save(agent.actor.state_dict(), ckpt_path)
            wandb.save(ckpt_path)

    # save final checkpoint + final obs_norm
    final_path = os.path.join(checkpoint_dir, "actor_final.pth")
    torch.save(agent.actor.state_dict(), final_path)
    wandb.save(final_path)

    if obs_normalizer is not None:
        norm_path = os.path.join(checkpoint_dir, "obs_norm.npz")
        _save_obs_norm_npz(norm_path, obs_normalizer)
        wandb.save(norm_path)

    # write to the legacy actor_best.pth / obs_norm_best.npz for compatibility:
    # prefer the last phase's 'late' best
    # otherwise choose the overall highest-score saved model
    try:
        last_phase_idx = int(phase_meta[-1]["idx"])
        preferred = phase_best.get((last_phase_idx, "late"), None)

        best_any = None
        candidates = [preferred] if (preferred and preferred.get("actor_path")) else []
        if not candidates:
            for tr in phase_best.values():
                if tr.get("actor_path") is None:
                    continue
                if best_any is None or float(tr.get("best_score", -1e9)) > float(best_any.get("best_score", -1e9)):
                    best_any = tr
            if best_any:
                candidates = [best_any]

        if candidates and candidates[0] and candidates[0].get("actor_path"):
            agent.actor.load_state_dict(torch.load(candidates[0]["actor_path"], map_location=agent.device))
            torch.save(agent.actor.state_dict(), legacy_best_actor)
            wandb.save(legacy_best_actor)

            if obs_normalizer is not None and candidates[0].get("norm_path") and os.path.exists(candidates[0]["norm_path"]):
                import shutil as _shutil
                _shutil.copyfile(candidates[0]["norm_path"], legacy_best_norm)
                wandb.save(legacy_best_norm)

    except Exception:
        pass

    # ensure legacy best files exist even if eval never ran.
    if not os.path.exists(legacy_best_actor):
        torch.save(agent.actor.state_dict(), legacy_best_actor)
        wandb.save(legacy_best_actor)
        if obs_normalizer is not None and not os.path.exists(legacy_best_norm):
            _save_obs_norm_npz(legacy_best_norm, obs_normalizer)
            wandb.save(legacy_best_norm)

    wandb.finish()


if __name__ == "__main__":
    train()
