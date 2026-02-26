# Overview
This folder contains the bash scripts and the python code for our programs for the day of the actual RL competition, which took place on 2026-02-26, from 10:00 to 22:00.

> Note that the authentication tokens are removed for privacy reasons.

---

# Steps
Given your script of choice (e.g. TQC), make it exectuable via:

```bash
chmod +x stay_alive_tqc.sh
```

Then, create a new _tmux_ session and enter it via:

```bash
tmux new -s comprl_tqc
```

There, execute the script to have it continuously participate in the competition, even if a restart happens (which happened 3 times during our competition):

```bash
./stay_alive_tqc.sh
```

Then, leave the tmux session by pressing `Ctrl + b` and then `d` on your keyboard to have it safely run in the background.

To dump the scrollback buffer or the current visible output of a tmux session without re-attaching, do the following:

```bash
tmux capture-pane -pt "session_name"
```