import os
import sys
from typing import List, Optional

import numpy as np
import torch

# Ensure project root (where "agent_1" and "hockey" live) is on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from comprl.client import Agent, launch_client  # type: ignore

from agent_1.td3_agent import TD3Agent
from agent_1.tqc_agent import TQCAgent
from agent_1.sac_agent import SACAgent
from agent_1.normalization import RunningMeanStd

STATE_DIM = 18
ACTION_DIM = 4


def load_obs_normalizer(npz_path: str) -> RunningMeanStd:
    """
    Load observation normalizer from .npz file.
    Matches the saving logic in agent_1/train.py
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Obs normalizer .npz not found: {npz_path}")

    data = np.load(npz_path)
    rms = RunningMeanStd(shape=(STATE_DIM,))
    rms.mean = data["mean"]
    rms.var = data["var"]
    rms.count = float(data["count"])
    if "clip" in data.files:
        rms.clip = float(data["clip"])
    return rms


class UnifiedRLAgent(Agent):
    """
    CompRL-compatible wrapper that can host TD3, TQC, or SAC.
    """

    def __init__(self, algo: str, model_path: str, norm_path: Optional[str]) -> None:
        super().__init__()

        algo_l = algo.strip().lower()
        if algo_l not in {"td3", "tqc", "sac"}:
            raise ValueError(f"Unknown algo '{algo}'. Expected one of: td3, tqc, sac")

        self.algo = algo_l

        # 1. Build the underlying RL agent
        if self.algo == "td3":
            self.agent = TD3Agent(
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                exploration_noise_type="gaussian",
            )
        elif self.algo == "tqc":
            self.agent = TQCAgent(
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
            )
        elif self.algo == "sac":
            self.agent = SACAgent(
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
            )
        else:
            raise RuntimeError("Invalid algo branch")

        # 2. Load actor weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model .pth not found: {model_path}")

        state_dict = torch.load(model_path, map_location=self.agent.device)
        # All three agents expose .actor
        self.agent.actor.load_state_dict(state_dict)
        self.agent.actor.eval()
        print(f"[UnifiedRLAgent] Loaded {self.algo.upper()} actor from: {model_path}")

        # 3. Load observation normalizer (optional)
        self.normalizer = None
        if norm_path:
            if os.path.exists(norm_path):
                self.normalizer = load_obs_normalizer(norm_path)
                print(f"[UnifiedRLAgent] Loaded obs normalizer from: {norm_path}")
            else:
                print(f"[UnifiedRLAgent] Warning: norm_path given but file not found: {norm_path}")

    # ===== CompRL callbacks =====

    def get_step(self, observation: List[float]) -> List[float]:
        """
        Called every env step. Must return an action as list[float].
        """
        obs = np.asarray(observation, dtype=np.float32)

        # Normalize if we have RMS
        if self.normalizer is not None:
            obs = self.normalizer.normalize_np(obs)

        # Deterministic behavior (no exploration noise) for all algorithms:
        action = self.agent.select_action(obs, noise=0.0)
        return np.asarray(action, dtype=np.float32).tolist()

    def on_start_game(self, game_id) -> None:
        print(f"[UnifiedRLAgent] Game started: {game_id}")

    def on_end_game(self, result: bool, stats: List[float]) -> None:
        outcome = "WON" if result else "LOST"
        if stats and len(stats) >= 2:
            print(f"[UnifiedRLAgent] Game ended. Result: {outcome} | Score: {stats[0]} - {stats[1]}")
        else:
            print(f"[UnifiedRLAgent] Game ended. Result: {outcome} | Stats: {stats}")


def initialize_agent(agent_args: List[str]) -> Agent:
    """
    Required by comprl.client.launch_client.
    Reads config from environment variables.
    """
    algo = os.environ.get("UNIFIED_ALGO", "").strip().lower()
    model_path = os.environ.get("UNIFIED_MODEL_PATH", "")
    norm_path = os.environ.get("UNIFIED_NORM_PATH", "")

    if not algo:
        raise RuntimeError("UNIFIED_ALGO not set in environment")
    if not model_path:
        raise RuntimeError("UNIFIED_MODEL_PATH not set in environment")

    # If norm_path is empty string, set to None
    final_norm_path = norm_path if norm_path else None

    return UnifiedRLAgent(algo=algo, model_path=model_path, norm_path=final_norm_path)


def main():
    print("=== CompRL unified client for TD3 / TQC / SAC ===")

    # 1) Get values from environment variables (set by your bash command)
    algo = os.environ.get("UNIFIED_ALGO", "").strip().lower()
    model_path = os.environ.get("UNIFIED_MODEL_PATH", "").strip()
    norm_path = os.environ.get("UNIFIED_NORM_PATH", "").strip()

    # 2) Server config from env (with defaults)
    url = os.environ.get("COMPRL_SERVER_URL", "comprl.cs.uni-tuebingen.de").strip()
    port = os.environ.get("COMPRL_SERVER_PORT", "65335").strip()
    token = os.environ.get("COMPRL_ACCESS_TOKEN", "").strip()

    # Validation
    if not algo or algo not in {"td3", "tqc", "sac"}:
        raise SystemExit(f"Error: UNIFIED_ALGO must be td3, tqc, or sac (got: '{algo}')")
    if not model_path:
        raise SystemExit("Error: UNIFIED_MODEL_PATH is missing.")
    if not token:
        raise SystemExit("Error: COMPRL_ACCESS_TOKEN is missing.")

    print("\nConnecting to CompRL server:")
    print(f"  URL   : {url}")
    print(f"  PORT  : {port}")
    print(f"  ALGO  : {algo.upper()}")
    print(f"  MODEL : {model_path}")
    print(f"  NORM  : {norm_path if norm_path else '<none>'}")

    # 3) Start client loop
    # launch_client will call initialize_agent() internally
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()