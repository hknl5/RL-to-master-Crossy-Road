"""
Train a PPO agent on CrossyRoad — optimised for CPU (no GPU needed).

Key improvements over v1:
  - n_steps 512→2048, batch_size 128→512  (8× bigger rollouts, richer gradients)
  - net_arch [128,128]→[256,256]           (more representational capacity)
  - linear LR decay 3e-4→3e-5             (coarse early, fine later)
  - ent_coef 0.02→0.01                    (less random noise at high scores)
  - 16 parallel envs                       (more diverse experience per update)
  - 20M default steps

Usage:
    python train.py                        # 20M steps, 16 envs
    python train.py --timesteps 10000000 --n-envs 8
    python train.py --continue-from models/best_model  # resume from checkpoint
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from crossy_env import CrossyRoadEnv

LOG_DIR   = "logs"
MODEL_DIR = "models"


def linear_lr(initial: float, final: float):
    """Return a schedule callable: LR decays linearly from initial→final."""
    def schedule(progress_remaining: float) -> float:
        return final + (initial - final) * progress_remaining
    return schedule


def main(total_timesteps: int = 20_000_000,
         n_envs: int = 16,
         continue_from: str = ""):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    vec_env  = make_vec_env(CrossyRoadEnv, n_envs=n_envs)
    eval_env = CrossyRoadEnv()

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(50_000 // n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(200_000 // n_envs, 1),
        save_path=MODEL_DIR,
        name_prefix="ppo_crossy",
    )

    if continue_from:
        print(f"Continuing from checkpoint: {continue_from}")
        model = PPO.load(
            continue_from,
            env=vec_env,
            # keep same hyperparams but override LR schedule for fine-tuning
            learning_rate=linear_lr(1e-4, 3e-5),
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=None,
            # --- Rollout ---
            n_steps=2048,         # 2048 × 16 envs = 32 768 transitions per update
            batch_size=512,
            n_epochs=10,
            # --- Optimisation ---
            learning_rate=linear_lr(3e-4, 3e-5),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,        # lower than before: less noise at high scores
            # --- Network ---
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    est_mins = total_timesteps / 2_500 / 60
    print(f"Training for {total_timesteps:,} steps on {n_envs} CPU envs.")
    print(f"Estimated time: ~{est_mins:.0f} minutes\n")
    print("Watch agent:     python watch.py  (after training)\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, checkpoint_cb],
        progress_bar=True,
        reset_num_timesteps=not bool(continue_from),
    )

    path = os.path.join(MODEL_DIR, "ppo_crossy_final")
    model.save(path)
    print(f"\nDone! Model saved to {path}.zip")
    print("Run:  python watch.py")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",     type=int, default=20_000_000)
    parser.add_argument("--n-envs",        type=int, default=16)
    parser.add_argument("--continue-from", type=str, default="",
                        help="Path to a .zip checkpoint to resume from")
    args = parser.parse_args()
    main(args.timesteps, args.n_envs, args.continue_from)
