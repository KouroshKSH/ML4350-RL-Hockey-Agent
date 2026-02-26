def save_hockey_pkl_as_gif(pkl_path, gif_path="game.gif", fps=50, rounds="all", max_steps=None, pause_sec_on_goal=2.0, dpi=120):
    import pickle, textwrap
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import imageio.v2 as imageio

    class _NumpyCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            if module.startswith("numpy._globals"):
                module = module.replace("numpy._globals", "numpy", 1)
            return super().find_class(module, name)

    with open(pkl_path, "rb") as f:
        try:
            data = pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            data = _NumpyCompatUnpickler(f).load()

    p1_name, p2_name = "Player 1", "Player 2"
    if isinstance(data, dict):
        if isinstance(data.get("user_names"), (list, tuple)) and len(data["user_names"]) >= 2:
            p1_name, p2_name = str(data["user_names"][0]), str(data["user_names"][1])
        elif isinstance(data.get("user_ids"), (list, tuple)) and len(data["user_ids"]) >= 2:
            p1_name, p2_name = str(data["user_ids"][0]), str(data["user_ids"][1])

    def is_shabin(name: str) -> bool:
        return "shabin" in str(name).lower()

    p1_color = "red" if is_shabin(p1_name) else "tab:blue"
    p2_color = "red" if is_shabin(p2_name) else "tab:green"

    def wrap2(s: str, width=14):
        lines = textwrap.wrap(str(s), width=width)
        if len(lines) <= 2:
            return "\n".join(lines) if lines else str(s)
        first = lines[0]
        second = " ".join(lines[1:])
        second = textwrap.shorten(second, width=width, placeholder="…")
        return first + "\n" + second

    p1_label = wrap2(p1_name, width=14)
    p2_label = wrap2(p2_name, width=14)

    # rounds list
    if isinstance(data, dict) and isinstance(data.get("rounds"), (list, tuple)) and len(data["rounds"]) > 0:
        rounds_list = list(data["rounds"])
        def get_obs_from_round(r):
            if isinstance(r, dict) and "observations" in r:
                return np.asarray(r["observations"])
            return None
    else:
        arr = np.asarray(data)
        if arr.ndim == 2 and arr.shape[1] in (16, 18):
            rounds_list = [{"observations": arr}]
            def get_obs_from_round(r): return np.asarray(r["observations"])
        else:
            raise ValueError("Could not find rounds or a (T,16/18) observation array in the pickle.")

    if rounds == "all":
        round_indices = list(range(len(rounds_list)))
    elif isinstance(rounds, int):
        round_indices = [rounds]
    elif isinstance(rounds, (list, tuple)):
        round_indices = list(rounds)
    else:
        raise ValueError('rounds must be "all", an int, or a list of ints.')

    # rink + goal geometry (approx)
    W, H = 10.0, 8.0
    xlim = (-W / 2, W / 2)
    ylim = (-H / 2, H / 2)

    SCALE = 60.0
    GOAL_SIZE = 75.0
    goal_half_w = (20.0 / SCALE) / 2.0
    goal_half_h = (2.0 * GOAL_SIZE / SCALE) / 2.0
    goal_x_left = -(245.0 / SCALE + 10.0 / SCALE)
    goal_x_right = +(245.0 / SCALE + 10.0 / SCALE)

    def puck_in_goal(puck_xy):
        x, y = float(puck_xy[0]), float(puck_xy[1])
        in_left = (abs(x - goal_x_left) <= goal_half_w) and (abs(y) <= goal_half_h)
        in_right = (abs(x - goal_x_right) <= goal_half_w) and (abs(y) <= goal_half_h)
        return in_left, in_right

    pause_frames = int(round(float(pause_sec_on_goal) * float(fps)))
    timeline = []
    per_round_obs = []
    round_winner_text = []

    for ri in round_indices:
        obs = get_obs_from_round(rounds_list[ri])
        if obs is None or obs.ndim != 2 or obs.shape[1] not in (16, 18):
            raise ValueError(f"Round {ri} observations missing or not shaped (T,16/18).")
        if max_steps is not None and obs.shape[0] > int(max_steps):
            obs = obs[: int(max_steps)]

        goal_t = None
        goal_side = None
        for t in range(obs.shape[0]):
            puck = obs[t][12:14]
            in_left, in_right = puck_in_goal(puck)
            if in_left or in_right:
                goal_t = t
                goal_side = "left" if in_left else "right"
                break

        per_round_obs.append(obs)
        local_i = len(per_round_obs) - 1

        if goal_side == "left":
            round_winner_text.append(f"{p2_name} wins")
        elif goal_side == "right":
            round_winner_text.append(f"{p1_name} wins")
        else:
            round_winner_text.append("Agents draw")

        T = obs.shape[0] if goal_t is None else (goal_t + 1)
        for t in range(T):
            timeline.append((local_i, t))
        if goal_t is not None and pause_frames > 0:
            for _ in range(pause_frames):
                timeline.append((local_i, goal_t))

    # figure
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=dpi)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    ax.plot([xlim[0], xlim[1], xlim[1], xlim[0], xlim[0]],
            [ylim[0], ylim[0], ylim[1], ylim[1], ylim[0]])
    ax.plot([0, 0], [ylim[0], ylim[1]], linestyle="--", linewidth=1)

    ax.add_patch(Rectangle((goal_x_left - goal_half_w, -goal_half_h),
                           2 * goal_half_w, 2 * goal_half_h, fill=False, linewidth=2))
    ax.add_patch(Rectangle((goal_x_right - goal_half_w, -goal_half_h),
                           2 * goal_half_w, 2 * goal_half_h, fill=False, linewidth=2))

    header = ax.text(
        0.5, 1.02, "", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75)
    )

    ax.text(0.01, 0.98, f"P1 (left): {p1_name}", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, color=p1_color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))
    ax.text(0.99, 0.98, f"P2 (right): {p2_name}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color=p2_color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))

    p1_dot, = ax.plot([], [], marker="o", markersize=6, linestyle="", color=p1_color)
    p2_dot, = ax.plot([], [], marker="o", markersize=6, linestyle="", color=p2_color)
    puck_dot, = ax.plot([], [], marker="o", markersize=5, linestyle="", color="black")

    box_w, box_h = 0.55, 0.55
    p1_box = Rectangle((0, 0), box_w, box_h, fill=False, linewidth=2, edgecolor=p1_color)
    p2_box = Rectangle((0, 0), box_w, box_h, fill=False, linewidth=2, edgecolor=p2_color)
    ax.add_patch(p1_box)
    ax.add_patch(p2_box)

    p1_text = ax.text(0, 0, p1_label, ha="center", va="bottom", fontsize=9, color=p1_color,
                      bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    p2_text = ax.text(0, 0, p2_label, ha="center", va="bottom", fontsize=9, color=p2_color,
                      bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    goal_text = ax.text(0, ylim[1] - 0.3, "", ha="center", va="top", fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))

    frames = []
    total_rounds = len(per_round_obs)

    for (local_round_i, t) in timeline:
        obs = per_round_obs[local_round_i][t]

        p1 = obs[0:2]
        p2 = obs[6:8]
        puck = obs[12:14]

        header_line1 = f"Round {local_round_i + 1}/{total_rounds}"
        header_line2 = textwrap.shorten(f"{p1_name} vs {p2_name}", width=55, placeholder="…")
        header_line3 = round_winner_text[local_round_i]
        header.set_text(header_line1 + "\n" + header_line2 + "\n" + header_line3)

        p1_dot.set_data([p1[0]], [p1[1]])
        p2_dot.set_data([p2[0]], [p2[1]])
        puck_dot.set_data([puck[0]], [puck[1]])

        p1_box.set_xy((float(p1[0]) - box_w / 2, float(p1[1]) - box_h / 2))
        p2_box.set_xy((float(p2[0]) - box_w / 2, float(p2[1]) - box_h / 2))

        p1_text.set_position((float(p1[0]), float(p1[1]) + box_h / 2 + 0.06))
        p2_text.set_position((float(p2[0]), float(p2[1]) + box_h / 2 + 0.06))

        in_left, in_right = puck_in_goal(puck)
        if in_left:
            goal_text.set_text("GOAL: Player 2 scores (Player 1 loses)")
        elif in_right:
            goal_text.set_text("GOAL: Player 1 scores (Player 2 loses)")
        else:
            goal_text.set_text("")

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())

    plt.close(fig)
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"Saved GIF to: {gif_path}")