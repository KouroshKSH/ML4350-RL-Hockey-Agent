import os
import argparse
import yaml
import wandb
from pathlib import Path
from datetime import datetime

# Abtin's renderer
from version2_gif import save_hockey_pkl_as_gif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("results", "gifs", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    tasks = []

    # Collect explicit pkl files
    if cfg.get("pkl_files"):
        tasks.extend(cfg["pkl_files"])

    # Collect from folders
    if cfg.get("folders"):
        for folder in cfg["folders"]:
            if os.path.isdir(folder):
                tasks.extend([
                    os.path.join(folder, f)
                    for f in os.listdir(folder)
                    if f.endswith(".pkl")
                ])

    if not tasks:
        print("No .pkl files found.")
        return

    wandb.init(
        project="hockey_visualizations",
        name=f"Bulk_Render_{timestamp}"
    )

    for pkl_path in tasks:
        print(f"Processing: {pkl_path}")
        try:
            gif_filename = f"{Path(pkl_path).stem}.gif"
            gif_path = os.path.join(output_dir, gif_filename)

            # Call your friend's working function
            save_hockey_pkl_as_gif(
                pkl_path=pkl_path,
                gif_path=gif_path,
                fps=50,
                rounds="all",
                max_steps=None,
                pause_sec_on_goal=2.0,
                dpi=120
            )

            # Log to W&B
            wandb.log({
                "game_gif": wandb.Video(gif_path, format="gif"),
                "source": str(pkl_path)
            })

            print(f"Successfully saved to {gif_path}")

        except Exception as e:
            print(f"Error rendering {pkl_path}: {e}")

    wandb.finish()


if __name__ == "__main__":
    main()
