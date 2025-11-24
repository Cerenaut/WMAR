import itertools
from pathlib import Path
from copy import deepcopy
from typing import Optional
import gc
import torch

from config import Config
from sac import train_sac


def run_grid(config_path: str, out_path: Optional[str] = None) -> None:
    base_cfg = Config.from_file(Path(config_path))

    # Define a small grid. Adjust as needed.
    grid = {
        # Collection schedule
        # Optimisation and stability
        "sac_lr": [3e-4, 1e-4],
        "sac_batch_size": [48, 32],
        "sac_target_entropy_coef": [1.0, 0.9],
        "sac_grad_clip": [5.0, 25.0],
    }

    keys = list(grid.keys())
    best_score = float("-inf")
    best_combo: dict[str, float | int] = {}

    total_combos = 1
    for k in keys:
        total_combos *= len(grid[k])
    print(f"Total combinations: {total_combos}")

    # Warn if replay is on GPU (grid search can be VRAM heavy)
    try:
        rb_dev = base_cfg.replay_buffers[0].rb_device
        if isinstance(rb_dev, str) and rb_dev.lower().startswith("cuda"):
            print("[WARN] Replay buffer is on CUDA in base config. Consider rb_device=\"cpu\" for grid search.")
    except Exception:
        pass

    # Prepare output file
    out_file = Path(out_path) if out_path is not None else Path.cwd() / "gridsearch_results.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        fp.write(f"Total combinations: {total_combos}\n")
        try:
            rb_dev = base_cfg.replay_buffers[0].rb_device
            if isinstance(rb_dev, str) and rb_dev.lower().startswith("cuda"):
                fp.write("[WARN] Replay buffer is on CUDA in base config. Consider rb_device=\"cpu\" for grid search.\n")
        except Exception:
            pass

    for values in itertools.product(*[grid[k] for k in keys]):
        cfg = deepcopy(base_cfg)
        # Short run for quick comparisons
        cfg.epochs = 10
        for k, v in zip(keys, values):
            setattr(cfg, k, v)

        # Keep constraint valid: n_sync * gen_seq_len == data_n * data_t
        # data_n and data_t are kept fixed in the grid; enforce the assertion here
        assert cfg.n_sync * cfg.gen_seq_len == cfg.data_n * cfg.data_t, (
            f"Constraint fail: {cfg.n_sync}*{cfg.gen_seq_len} != {cfg.data_n}*{cfg.data_t}"
        )

        combo = {k: getattr(cfg, k) for k in keys}
        print(f"\n=== Running combo: {combo} ===")
        with open(out_file, "a") as fp:
            fp.write(f"RUN {combo}\n")
        score = train_sac(cfg)
        # GPU memory cleanup between runs
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass
        print(f"=== Combo score: {score:.3f} ===\n")
        with open(out_file, "a") as fp:
            fp.write(f"SCORE {score:.6f}\n\n")
        if score > best_score:
            best_score = score
            best_combo = combo

    print("Best combo:")
    print(best_combo)
    print(f"Best score: {best_score:.3f}")
    with open(out_file, "a") as fp:
        fp.write("BEST\n")
        fp.write(f"COMBO {best_combo}\n")
        fp.write(f"SCORE {best_score:.6f}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to base config JSON")
    parser.add_argument("--out", required=False, help="Output text file path for logging results")
    args = parser.parse_args()
    run_grid(args.config, args.out)


