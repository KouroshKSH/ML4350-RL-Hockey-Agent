import random
import math
import inspect
import numpy as np
import torch

import hockey.hockey_env as h_env
from .external_models import discover_external_models

class SelfPlayPool:
    """
    Snapshot pool + lightweight performance stats.

    Supports multiple sampling strategies:
      - "ucb": difficulty (1 - win_score) + UCB bonus
      - "pfsp": proportional to (1 - win_score)^p  (focus weaknesses)
      - "pfsp_mid": focus opponents near target win_score (default target=0.5)
      - "meta": use externally provided meta_probs (e.g., PSRO)

    Extra option: discounted statistics (Discounted-UCB style).
      If discount_factor < 1, we exponentially discount all snapshot stats once per episode.
      This makes "hard" opponents track your *current* weaknesses instead of old history.
    """

    def __init__(
        self,
        max_size=20,
        elite_size=5,
        uniform_prob=0.10,
        ucb_beta=0.50,
        discount_stats: bool = False,
        discount_factor: float = 1.0,
    ):
        self.max_size = int(max_size)
        self.elite_size = int(elite_size)
        self.uniform_prob = float(uniform_prob)
        self.ucb_beta = float(ucb_beta)

        self.discount_stats = bool(discount_stats)
        self.discount_factor = float(discount_factor)

        self.snapshots = []
        self._next_id = 0
        # With discounting enabled, this becomes a float "effective" count.
        self.total_games = 0.0

    def __len__(self):
        return len(self.snapshots)

    def ids(self):
        return [int(e["id"]) for e in self.snapshots]

    def _new_id(self):
        sid = self._next_id
        self._next_id += 1
        return sid

    def _filter_actor_kwargs(self, actor_cls, actor_init_kwargs: dict) -> dict:
        """Prevent crashes when config passes kwargs that the actor class does not accept."""
        try:
            sig = inspect.signature(actor_cls.__init__)
            allowed = set(sig.parameters.keys())
            allowed.discard("self")
            return {k: v for k, v in (actor_init_kwargs or {}).items() if k in allowed}
        except Exception:
            return dict(actor_init_kwargs or {})

    def add_snapshot(self, actor_module, actor_init_kwargs, strength=None):
        actor_cls = actor_module.__class__
        filtered_kwargs = self._filter_actor_kwargs(actor_cls, actor_init_kwargs)

        sd = {k: v.detach().cpu().clone() for k, v in actor_module.state_dict().items()}
        entry = {
            "id": self._new_id(),
            "state_dict": sd,
            "actor_cls": actor_cls,
            "actor_kwargs": dict(filtered_kwargs),
            "has_sample": hasattr(actor_module, "sample"),

            # optional quality at creation time (used for elite retention)
            "strength": None if strength is None else float(strength),

            # online stats (float so discounting works)
            "games": 0.0,
            "agent_wins": 0.0,
            "ties": 0.0,
            "agent_losses": 0.0,
        }
        self.snapshots.append(entry)
        self._trim()
        return entry

    def add_preloaded(self, actor_cls, actor_kwargs: dict, state_dict: dict, has_sample: bool,
                    strength=None, obs_norm=None, meta: dict | None = None):
        filtered_kwargs = self._filter_actor_kwargs(actor_cls, actor_kwargs)
        entry = {
            "id": self._new_id(),
            "state_dict": state_dict,
            "actor_cls": actor_cls,
            "actor_kwargs": dict(filtered_kwargs),
            "has_sample": bool(has_sample),
            "strength": None if strength is None else float(strength),
            "obs_norm": obs_norm,       # RunningMeanStd or None
            "meta": dict(meta or {}),   # any extra info

            "games": 0.0,
            "agent_wins": 0.0,
            "ties": 0.0,
            "agent_losses": 0.0,
        }
        self.snapshots.append(entry)
        self._trim()
        return entry


    def _elite_ids(self):
        scored = [(e["strength"], e["id"]) for e in self.snapshots if e.get("strength") is not None]
        if not scored or self.elite_size <= 0:
            return set()
        scored.sort(key=lambda x: x[0], reverse=True)
        return set([sid for _s, sid in scored[: self.elite_size]])

    def _trim(self):
        if len(self.snapshots) <= self.max_size:
            return
        elite = self._elite_ids()
        kept = list(self.snapshots)

        while len(kept) > self.max_size:
            drop_idx = None
            for i, e in enumerate(kept):
                if e["id"] not in elite:
                    drop_idx = i
                    break
            if drop_idx is None:
                drop_idx = 0
            kept.pop(drop_idx)

        self.snapshots = kept

    def apply_discount(self):
        """Apply exponential discount to all stats once per episode (cheap: pool is small)."""
        if not self.discount_stats:
            return
        g = float(self.discount_factor)
        if g >= 1.0:
            return
        for e in self.snapshots:
            e["games"] *= g
            e["agent_wins"] *= g
            e["ties"] *= g
            e["agent_losses"] *= g
        self.total_games *= g

    def record_result(self, snapshot_id, winner):
        """
        winner is env winner from our perspective:
          1 = we win, 0 = tie, -1 = we lose
        """
        e = next((x for x in self.snapshots if x["id"] == snapshot_id), None)
        if e is None:
            return
        e["games"] += 1.0
        self.total_games += 1.0
        if winner == 1:
            e["agent_wins"] += 1.0
        elif winner == 0:
            e["ties"] += 1.0
        else:
            e["agent_losses"] += 1.0

    def _win_score(self, e) -> float:
        """Score in [0, 1]: win=1, tie=0.5, loss=0"""
        g = float(e.get("games", 0.0))
        if g <= 1e-9:
            return 0.5
        return float((e["agent_wins"] + 0.5 * e["ties"]) / g)

    def sample(
        self,
        strategy: str = "ucb",
        pfsp_power: float = 1.0,
        pfsp_target: float = 0.5,
        pfsp_temperature: float = 0.15,
        meta_probs: dict | None = None,
    ):
        if not self.snapshots:
            return None

        strategy = str(strategy or "ucb").lower()
        n = len(self.snapshots)

        # Strategy: externally provided meta distribution, a type of PSRO
        if strategy in ("meta", "psro"):
            if not meta_probs:
                # fallback to pfsp_mid
                strategy = "pfsp_mid"
            else:
                weights = []
                for e in self.snapshots:
                    w = float(meta_probs.get(int(e["id"]), 0.0))
                    weights.append(max(0.0, w))
                s = float(sum(weights))
                if s <= 0:
                    return random.choice(self.snapshots)
                weights = [(1.0 - self.uniform_prob) * (w / s) + self.uniform_prob * (1.0 / n) for w in weights]
                return random.choices(self.snapshots, weights=weights, k=1)[0]

        # Otherwise compute weights from stats
        weights = []

        if strategy == "pfsp":
            p = max(0.0, float(pfsp_power))
            for e in self.snapshots:
                score = self._win_score(e)
                w = (1.0 - score) ** p
                weights.append(max(1e-6, float(w)))

        elif strategy in ("pfsp_mid", "pfsp_target"):
            # Prefer opponents where score is near target (often ~0.5)
            t = float(pfsp_target)
            temp = max(1e-6, float(pfsp_temperature))
            for e in self.snapshots:
                score = self._win_score(e)
                z = (score - t) / temp
                w = math.exp(-0.5 * z * z)
                weights.append(max(1e-6, float(w)))

        else:  # "ucb"
            # With discounting enabled, total_games and games are "effective" counts.
            tg = float(self.total_games)
            for e in self.snapshots:
                score = self._win_score(e)
                diff = 1.0 - score
                g = float(e.get("games", 0.0))
                bonus = self.ucb_beta * math.sqrt(math.log(tg + 1.0) / (g + 1.0))
                w = max(1e-6, float(diff + bonus))
                weights.append(w)

        s = float(sum(weights))
        if s <= 0:
            return random.choice(self.snapshots)

        probs = [(1.0 - self.uniform_prob) * (w / s) + self.uniform_prob * (1.0 / n) for w in weights]
        return random.choices(self.snapshots, weights=probs, k=1)[0]


class OpponentManager:
    """
    - Samples opponent per episode from a mixture.
    - Curriculum: ramp in self-play after snapshot_start_episode.
    - Self-play: sample a SINGLE snapshot per episode (ctx).
    - Sampling: configurable ("ucb", "pfsp", "pfsp_mid", "meta").
    - Hygiene: bounded pool + elite retention by snapshot "strength".
    - Possible: discounted self-play stats (Discounted-UCB style).
    """

    def __init__(self, cfg, state_dim, action_dim, device):
        self.cfg = cfg
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = device

        self.weak = h_env.BasicOpponent(weak=True)
        self.strong = h_env.BasicOpponent(weak=False)

        mix = (cfg.get("opponent_mix") or {"weak": 0.35, "strong": 0.35, "self_play": 0.30})
        self.p_weak = float(mix.get("weak", 0.35))
        self.p_strong = float(mix.get("strong", 0.35))
        self.p_self = float(mix.get("self_play", 0.30))

        sp_cfg = (cfg.get("self_play") or {})
        self.self_play_enabled = bool(sp_cfg.get("enabled", True))
        self.snapshot_interval = int(sp_cfg.get("snapshot_interval_episodes", 200))
        self.snapshot_min_episode = int(sp_cfg.get("snapshot_start_episode", 400))
        self.deterministic_opponent = bool(sp_cfg.get("deterministic_opponent", True))

        pool_max = int(sp_cfg.get("max_pool_size", 20))
        elite_size = int(sp_cfg.get("elite_size", 5))
        uniform_prob = float(sp_cfg.get("sample_uniform_prob", 0.10))
        ucb_beta = float(sp_cfg.get("ucb_beta", 0.50))

        # Discounted-UCB style recency weighting
        discount_stats = bool(sp_cfg.get("discount_stats", False))
        discount_factor = float(sp_cfg.get("discount_factor", 1.0))

        self.pool = SelfPlayPool(
            max_size=pool_max,
            elite_size=elite_size,
            uniform_prob=uniform_prob,
            ucb_beta=ucb_beta,
            discount_stats=discount_stats,
            discount_factor=discount_factor,
        )

        extra_kwargs = sp_cfg.get("actor_kwargs", {}) or {}
        self.actor_init_kwargs = {"state_dim": self.state_dim, "action_dim": self.action_dim, **extra_kwargs}

        # sampling strategy
        self.self_play_sampling = str(sp_cfg.get("sampling", "pfsp_mid")).lower()
        self.pfsp_power = float(sp_cfg.get("pfsp_power", 1.0))
        self.pfsp_target = float(sp_cfg.get("pfsp_target", 0.5))
        self.pfsp_temperature = float(sp_cfg.get("pfsp_temperature", 0.15))

        self.meta_probs = None  # dict snapshot_id -> prob

        cur_cfg = (cfg.get("curriculum") or {})
        self.curriculum_enabled = bool(cur_cfg.get("enabled", True))
        self.self_play_ramp_episodes = int(cur_cfg.get("self_play_ramp_episodes", 1000))


        mix = (cfg.get("opponent_mix") or {})
        self.p_external = float(mix.get("external", 0.0))

        ext_cfg = (cfg.get("external_pool") or {})
        self.external_enabled = bool(ext_cfg.get("enabled", False))

        self.external_follow_self_sampling = bool(ext_cfg.get("follow_self_play_sampling", True))
        self.external_deterministic = bool(ext_cfg.get("deterministic_opponent", True))

        self.external_sampling = str(ext_cfg.get("sampling", self.self_play_sampling)).lower()
        self.ext_pfsp_power = float(ext_cfg.get("pfsp_power", self.pfsp_power))
        self.ext_pfsp_target = float(ext_cfg.get("pfsp_target", self.pfsp_target))
        self.ext_pfsp_temperature = float(ext_cfg.get("pfsp_temperature", self.pfsp_temperature))

        ext_pool_max = int(ext_cfg.get("max_pool_size", 200))
        ext_elite = int(ext_cfg.get("elite_size", 0))
        ext_uniform = float(ext_cfg.get("sample_uniform_prob", 0.10))
        ext_ucb_beta = float(ext_cfg.get("ucb_beta", 0.50))
        ext_discount_stats = bool(ext_cfg.get("discount_stats", False))
        ext_discount_factor = float(ext_cfg.get("discount_factor", 1.0))

        self.external_pool = SelfPlayPool(
            max_size=ext_pool_max,
            elite_size=ext_elite,
            uniform_prob=ext_uniform,
            ucb_beta=ext_ucb_beta,
            discount_stats=ext_discount_stats,
            discount_factor=ext_discount_factor,
        )

        # curriculum knobs for external
        self.external_start_episode = int(ext_cfg.get("start_episode", 0))
        self.external_ramp_episodes = int(ext_cfg.get("ramp_episodes", 0))

        # discover + preload
        if self.external_enabled:
            models = discover_external_models(cfg, state_dim=self.state_dim, action_dim=self.action_dim)
            for m in models:
                self.external_pool.add_preloaded(
                    actor_cls=m["actor_cls"],
                    actor_kwargs=m["actor_kwargs"],
                    state_dict=m["state_dict"],
                    has_sample=m["has_sample"],
                    strength=m.get("strength", None),
                    obs_norm=m.get("obs_norm", None),
                    meta={"algo": m.get("algo"), "run_dir": m.get("run_dir"), "actor_path": m.get("actor_path")},
                )
            if len(self.external_pool) == 0:
                raise RuntimeError(
                    "external_pool.enabled=true but no external models were loaded. "
                    "Check external_pool.results_root / env_name / score_threshold / algos / include_segments."
                )



    def set_self_play_sampling(self, strategy: str):
        self.self_play_sampling = str(strategy or self.self_play_sampling).lower()
        if getattr(self, "external_follow_self_sampling", False):
            self.external_sampling = self.self_play_sampling


    def set_meta_probs(self, meta_probs: dict | None):
        self.meta_probs = None if not meta_probs else dict(meta_probs)

    def maybe_add_snapshot(self, episode, agent, strength=None):
        if not self.self_play_enabled:
            return None
        if episode < self.snapshot_min_episode:
            return None
        if episode % self.snapshot_interval != 0:
            return None
        if hasattr(agent, "actor"):
            actor_kwargs = getattr(agent, "actor_init_kwargs", None) or self.actor_init_kwargs
            return self.pool.add_snapshot(agent.actor, actor_init_kwargs=actor_kwargs, strength=strength)
        return None

    def start_episode(self, kind, obs_normalizer=None):
        if kind in ("weak", "strong"):
            return kind, None

        if kind == "external":
            if not self.external_enabled:
                raise RuntimeError("start_episode('external') called but external_pool.enabled is false.")
            if len(self.external_pool) == 0:
                raise RuntimeError(
                    "external_pool.enabled=true but external_pool is empty. "
                    "This should have failed in OpponentManager.__init__."
                )

            entry = self.external_pool.sample(
                strategy=self.external_sampling,
                pfsp_power=self.ext_pfsp_power,
                pfsp_target=self.ext_pfsp_target,
                pfsp_temperature=self.ext_pfsp_temperature,
                meta_probs=None,
            )
            if entry is None:
                raise RuntimeError("external_pool.sample(...) returned None (unexpected).")

            actor = entry["actor_cls"](**entry["actor_kwargs"]).to(self.device)
            actor.load_state_dict(entry["state_dict"], strict=True)
            actor.eval()

            ctx = {
                "actor": actor,
                "has_sample": bool(entry.get("has_sample", False)),
                "snapshot_id": int(entry["id"]),
                "pool": "external",
                "obs_norm": entry.get("obs_norm", None),
            }
            return "external", ctx

        # self_play
        if (not self.self_play_enabled) or (len(self.pool) == 0):
            return "strong", None

        entry = self.pool.sample(
            strategy=self.self_play_sampling,
            pfsp_power=self.pfsp_power,
            pfsp_target=self.pfsp_target,
            pfsp_temperature=self.pfsp_temperature,
            meta_probs=self.meta_probs,
        )
        if entry is None:
            return "strong", None

        actor = entry["actor_cls"](**entry["actor_kwargs"]).to(self.device)
        actor.load_state_dict(entry["state_dict"], strict=True)
        actor.eval()

        ctx = {
            "actor": actor,
            "has_sample": bool(entry.get("has_sample", False)),
            "snapshot_id": int(entry["id"]),
            "pool": "self_play",
            "obs_norm": obs_normalizer,
        }
        return "self_play", ctx


    @torch.no_grad()
    def act(self, kind, obs_agent2, ctx=None):
        if kind == "weak":
            return self.weak.act(obs_agent2)
        if kind == "strong":
            return self.strong.act(obs_agent2)

        if kind in ("self_play", "external") and (ctx is None or "actor" not in ctx):
            raise RuntimeError(f"act called for kind='{kind}' but ctx/actor missing.")

        actor = ctx["actor"]

        obs_use = np.asarray(obs_agent2, dtype=np.float32)
        s = torch.as_tensor(obs_use.reshape(1, -1), device=self.device)

        has_sample = bool(ctx.get("has_sample", False))
        if has_sample:
            deterministic = self.deterministic_opponent if ctx.get("pool") == "self_play" else self.external_deterministic
            if deterministic:
                _, _, a = actor.sample(s)
            else:
                a, _, _ = actor.sample(s)
            return a.cpu().numpy().flatten()

        a = actor(s)
        return a.cpu().numpy().flatten()

    def end_episode(self, kind, ctx, winner):
        self.pool.apply_discount()
        if getattr(self, "external_pool", None) is not None:
            self.external_pool.apply_discount()

        if ctx is None:
            return
        sid = ctx.get("snapshot_id", None)
        if sid is None:
            return

        if kind == "self_play":
            self.pool.record_result(int(sid), int(winner))
        elif kind == "external":
            self.external_pool.record_result(int(sid), int(winner))
