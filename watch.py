"""
Watch a trained agent play CrossyRoad.

Usage:
    python watch.py                          # loads models/best_model
    python watch.py models/ppo_crossy_final  # load a specific checkpoint
"""
import sys
import pygame
from stable_baselines3 import PPO
from crossy_env import CrossyRoadEnv


def main(model_path: str = "models/best_model"):
    env = CrossyRoadEnv(render_mode="human", max_steps=2000)

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    env.render()  # opens the window before the event loop starts
    episode = 1
    total_reward = 0.0

    print("Close the window to stop.\n")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode {episode:3d}  |  score: {info['score']:5d}  |  reward: {total_reward:7.2f}")
            obs, _ = env.reset()
            episode += 1
            total_reward = 0.0

        pygame.time.wait(80)  # slow down so it's watchable (~12 fps)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "models/best_model"
    main(path)
