"""
Human play — REAL-TIME.

Cars move automatically every tick even if you don't press anything.
You WILL die standing still on a road.

Controls:  Arrow keys or WASD   |   Q / Esc to quit
"""
import sys
import pygame
from crossy_env import CrossyRoadEnv, UP, DOWN, LEFT, RIGHT, STAY

KEY_MAP = {
    pygame.K_UP:    UP,
    pygame.K_w:     UP,
    pygame.K_DOWN:  DOWN,
    pygame.K_s:     DOWN,
    pygame.K_LEFT:  LEFT,
    pygame.K_a:     LEFT,
    pygame.K_RIGHT: RIGHT,
    pygame.K_d:     RIGHT,
}

# How fast the game ticks (ms). Lower = harder.
# 120 ms ≈ 8 game-ticks / second.  Speed-1 cars cross 11 cells in ~1.3 s.
TICK_MS = 120

GAME_TICK = pygame.USEREVENT + 1


def main():
    env = CrossyRoadEnv(render_mode="human", max_steps=10_000)
    obs, _ = env.reset()
    env.render()   # opens the window

    pygame.time.set_timer(GAME_TICK, TICK_MS)

    queued_action = STAY   # default: stand still (cars still advance!)
    waiting_restart = False

    print(f"Tick: {TICK_MS} ms  |  Arrow / WASD to move  |  Q/Esc to quit")

    while True:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                env.close()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    env.close()
                    sys.exit()

                if waiting_restart:
                    # Any key restarts after death
                    obs, _ = env.reset()
                    waiting_restart = False
                    queued_action = STAY
                    continue

                action = KEY_MAP.get(event.key)
                if action is not None:
                    queued_action = action

            elif event.type == GAME_TICK:
                if waiting_restart:
                    continue

                obs, reward, terminated, truncated, info = env.step(queued_action)
                queued_action = STAY   # consume — next tick defaults to STAY

                if terminated:
                    print(f"Dead!  Score: {info['score']}  — press any key to restart")
                    waiting_restart = True
                elif truncated:
                    print(f"Time up!  Score: {info['score']}  — press any key to restart")
                    waiting_restart = True

        # Small sleep keeps the event loop responsive between ticks
        pygame.time.wait(8)


if __name__ == "__main__":
    main()
