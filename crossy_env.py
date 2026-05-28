"""
Crossy Road — Gymnasium environment, Pygame rendering.
Compatible with Stable-Baselines3.

Actions: UP=0  DOWN=1  LEFT=2  RIGHT=3  STAY=4
Cars/logs advance every tick regardless of player action.

Row types: GRASS (safe), ROAD (cars — deadly), WATER (logs — deadly if no log)
"""
import gymnasium as gym
import numpy as np
import pygame
from typing import Optional

# ---------------------------------------------------------------------------
GRID_COLS        = 11
VISIBLE_ROWS     = 10
PLAYER_START_COL = GRID_COLS // 2
TILE_SIZE = 64
SCREEN_W  = GRID_COLS * TILE_SIZE
SCREEN_H  = VISIBLE_ROWS * TILE_SIZE

SCROLL_THRESHOLD   = 3
AUTOSCROLL_START   = 28
AUTOSCROLL_MIN     = 8

GRASS, ROAD, WATER = 0, 1, 2
UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C_GRASS_A  = ( 34, 120,  34)
C_GRASS_B  = ( 44, 148,  44)
C_ROAD     = ( 55,  55,  55)
C_KERB     = (200, 180,  60)
C_MARK     = (200, 200,  80)
C_CAR_FAST = (220,  40,  40)
C_CAR_SLOW = (210, 120,  30)
C_WIN      = (160, 200, 255)
C_WATER_A  = ( 30, 100, 200)
C_WATER_B  = ( 20,  80, 180)
C_LOG      = (139,  90,  43)
C_LOG_RING = (110,  70,  30)
C_PLAYER   = (255, 220,  50)
C_BEAK     = (255, 150,   0)
C_EYE      = ( 25,  25,  25)
C_SHINE    = (255, 255, 255)
C_HUD_BG   = (  0,   0,   0)
C_HUD_TXT  = (255, 255, 255)
C_DEAD_TXT = (255,  50,  50)

# ---------------------------------------------------------------------------
# Observation cell encoding (distinct, evenly-spaced bands)
# ---------------------------------------------------------------------------
#   0.00  grass
#   0.10  water (deadly)
#   0.22  log moving right (safe)
#   0.32  log moving left  (safe)
#   0.45  empty road
#   0.55  car right fast   (deadly)
#   0.63  car right slow   (deadly)
#   0.73  car left  fast   (deadly)
#   0.82  car left  slow   (deadly)
#   1.00  player


class CrossyRoadEnv(gym.Env):
    """
    Observation  : flat float32 (224,)
        current grid (110) + 1-step-ahead danger/log grid (110) + 4 scalars
        scalars: player_row_norm, scroll_progress, difficulty, is_riding_log

    Reward       : +1 per new furthest row (high-water mark)
                   +0.05/step on grass
                   +0.02/step on log (water row)
                   -0.005/step on road
                   -1 on death
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 2000):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps   = max_steps

        self.action_space = gym.spaces.Discrete(5)
        # current grid (110) + lookahead grid (110) + 4 scalars = 224
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(VISIBLE_ROWS * GRID_COLS * 2 + 4,),
            dtype=np.float32,
        )

        self._window: Optional[pygame.Surface] = None
        self._clock:  Optional[pygame.time.Clock] = None
        self._font:   Optional[pygame.font.Font]  = None

        self._world:        list[dict] = []
        self._next_section: list[dict] = []
        self._player_col = 0
        self._player_row = 0
        self.score    = 0
        self._steps   = 0
        self._dead    = False
        self._rng     = np.random.default_rng()

    # -----------------------------------------------------------------------
    # World generation
    # -----------------------------------------------------------------------

    def _difficulty(self) -> float:
        return min(1.0, self.score / 200.0)

    def _make_grass_row(self) -> dict:
        return {"type": GRASS, "cars": [], "logs": []}

    def _make_road_row(self, direction: Optional[int] = None) -> dict:
        if direction is None:
            direction = 1 if self._rng.random() < 0.5 else -1

        d = self._difficulty()
        fast_prob = 0.05 + 0.80 * d
        speed = 1 if self._rng.random() < fast_prob else 2

        min_cars = 2
        max_cars = min(int(2 + round(3 * d)), GRID_COLS - 3)
        n = int(self._rng.integers(min_cars, max_cars + 1))

        cols = self._rng.choice(GRID_COLS, size=n, replace=False).tolist()
        cars = [
            {"col": int(c), "dir": direction, "speed": speed,
             "tick": int(self._rng.integers(0, speed))}
            for c in cols
        ]
        return {"type": ROAD, "cars": cars, "logs": []}

    def _make_water_row(self, direction: Optional[int] = None) -> dict:
        """Water row with floating logs. All logs share direction and speed."""
        if direction is None:
            direction = 1 if self._rng.random() < 0.5 else -1

        # Logs are always slow (speed=2) so the player has time to react
        speed = 2
        logs = []

        # Place logs left-to-right with random gaps between them
        col = int(self._rng.integers(0, 3))
        while col < GRID_COLS:
            length = int(self._rng.integers(2, 4))   # 2- or 3-cell wide log
            if col + length > GRID_COLS:
                break
            logs.append({
                "col":    col,
                "length": length,
                "dir":    direction,
                "speed":  speed,
                "tick":   int(self._rng.integers(0, speed)),
            })
            col += length + int(self._rng.integers(1, 4))  # 1–3 cell gap

        # Ensure at least one log so the row is crossable
        if not logs:
            logs.append({
                "col":    int(self._rng.integers(0, GRID_COLS - 2)),
                "length": 2,
                "dir":    direction,
                "speed":  speed,
                "tick":   0,
            })

        return {"type": WATER, "cars": [], "logs": logs}

    def _generate_section(self) -> list[dict]:
        d = self._difficulty()

        grass_prob = max(0.20, 0.60 - 0.40 * d)
        # Water visible from score 0; ramps from 20% → 30% at max difficulty
        water_prob = min(0.30, 0.20 + 0.10 * d)

        r = self._rng.random()

        if r < grass_prob:
            max_grass = max(1, int(round(3 - 2 * d)))
            n = int(self._rng.integers(1, max_grass + 1))
            return [self._make_grass_row() for _ in range(n)]

        elif r < grass_prob + water_prob:
            n = int(self._rng.integers(1, 3))   # 1–2 water rows
            direction = 1 if self._rng.random() < 0.5 else -1
            return [self._make_water_row(direction) for _ in range(n)]

        else:
            max_lanes = max(1, int(round(1 + 2 * d)))
            n = int(self._rng.integers(1, max_lanes + 1))
            direction = 1 if self._rng.random() < 0.5 else -1
            return [self._make_road_row(direction) for _ in range(n)]

    def _pop_next_row(self) -> dict:
        if not self._next_section:
            self._next_section = self._generate_section()
        return self._next_section.pop(0)

    # -----------------------------------------------------------------------
    # Collision helpers
    # -----------------------------------------------------------------------

    def _car_at(self, row_idx: int, col: int) -> bool:
        for car in self._world[row_idx]["cars"]:
            if car["col"] == col:
                return True
        return False

    def _log_at(self, row_idx: int, col: int) -> Optional[dict]:
        """Return the log occupying (row_idx, col), or None."""
        for log in self._world[row_idx]["logs"]:
            for i in range(log["length"]):
                if (log["col"] + i) % GRID_COLS == col:
                    return log
        return None

    def _on_log(self) -> bool:
        row = self._world[self._player_row]
        return row["type"] == WATER and self._log_at(self._player_row, self._player_col) is not None

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------

    def _get_future_grid(self, t: int) -> np.ndarray:
        """
        Lookahead grid (t steps ahead):
          1.0 = car will be here, or exposed water (no log)
          0.5 = log will be here (safe)
          0.0 = safe (grass or empty road)
        """
        grid = np.zeros((VISIBLE_ROWS, GRID_COLS), dtype=np.float32)

        for r, row in enumerate(self._world):
            if row["type"] == ROAD:
                for car in row["cars"]:
                    tick, col = car["tick"], car["col"]
                    for _ in range(t):
                        tick += 1
                        if tick >= car["speed"]:
                            tick = 0
                            col = (col + car["dir"]) % GRID_COLS
                    grid[r, col] = 1.0

            elif row["type"] == WATER:
                grid[r, :] = 1.0  # water is dangerous by default
                for log in row["logs"]:
                    tick, col = log["tick"], log["col"]
                    for _ in range(t):
                        tick += 1
                        if tick >= log["speed"]:
                            tick = 0
                            col = (col + log["dir"]) % GRID_COLS
                    for i in range(log["length"]):
                        c = (col + i) % GRID_COLS
                        grid[r, c] = 0.5  # log: safe

        return grid

    def _get_obs(self) -> np.ndarray:
        grid = np.zeros((VISIBLE_ROWS, GRID_COLS), dtype=np.float32)

        for r, row in enumerate(self._world):
            if row["type"] == ROAD:
                grid[r, :] = 0.45   # empty road
                for car in row["cars"]:
                    if car["dir"] == 1:   # right
                        grid[r, car["col"]] = 0.55 if car["speed"] == 1 else 0.63
                    else:                 # left
                        grid[r, car["col"]] = 0.73 if car["speed"] == 1 else 0.82

            elif row["type"] == WATER:
                grid[r, :] = 0.10   # deadly water background
                for log in row["logs"]:
                    val = 0.22 if log["dir"] == 1 else 0.32
                    for i in range(log["length"]):
                        c = (log["col"] + i) % GRID_COLS
                        grid[r, c] = val

        grid[self._player_row, self._player_col] = 1.0

        future = self._get_future_grid(1)

        interval = max(AUTOSCROLL_MIN,
                       int(AUTOSCROLL_START - (AUTOSCROLL_START - AUTOSCROLL_MIN)
                           * self._difficulty()))
        extras = np.array([
            self._player_row / (VISIBLE_ROWS - 1),
            self._scroll_tick / interval,
            self._difficulty(),
            1.0 if self._on_log() else 0.0,
        ], dtype=np.float32)

        return np.concatenate([grid.flatten(), future.flatten(), extras])

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.score        = 0
        self._steps       = 0
        self._dead        = False
        self._net_pos     = 0
        self._scroll_tick = 0
        self._player_col  = PLAYER_START_COL
        self._player_row  = VISIBLE_ROWS - 1
        self._next_section = []

        upper: list[dict] = []
        while len(upper) < VISIBLE_ROWS - 3:
            upper.extend(self._generate_section())
        upper = upper[:VISIBLE_ROWS - 3]

        self._world = upper + [self._make_grass_row() for _ in range(3)]
        return self._get_obs(), {}

    def step(self, action: int):
        self._steps += 1

        # Base per-step reward by terrain type
        row_type = self._world[self._player_row]["type"]
        if row_type == GRASS:
            reward = 0.05
        elif row_type == WATER:
            reward = 0.02   # riding a log — small survival bonus
        else:
            reward = -0.005  # road — nudge agent to cross promptly

        terminated = False

        # --- 1. Advance all cars -------------------------------------------
        for row in self._world:
            for car in row["cars"]:
                car["tick"] += 1
                if car["tick"] >= car["speed"]:
                    car["tick"] = 0
                    car["col"] = (car["col"] + car["dir"]) % GRID_COLS

        # --- 2. Advance logs; carry player if currently riding ---------------
        # Determine riding BEFORE logs move
        riding = None
        if self._world[self._player_row]["type"] == WATER:
            riding = self._log_at(self._player_row, self._player_col)

        for row in self._world:
            for log in row["logs"]:
                log["tick"] += 1
                if log["tick"] >= log["speed"]:
                    log["tick"] = 0
                    log["col"] = (log["col"] + log["dir"]) % GRID_COLS
                    if riding is not None and log is riding:
                        new_pc = self._player_col + log["dir"]
                        if new_pc < 0 or new_pc >= GRID_COLS:
                            # Log carried player off screen
                            terminated = True
                            reward = -1.0
                            self._dead = True
                        else:
                            self._player_col = new_pc

        # --- 3. Move player -------------------------------------------------
        if not terminated:
            nr = self._player_row
            nc = self._player_col

            if action == UP:
                nr -= 1
            elif action == DOWN:
                nr += 1
            elif action == LEFT:
                nc -= 1
            elif action == RIGHT:
                nc += 1

            nc = max(0, min(GRID_COLS - 1, nc))

            if action == UP:
                self._net_pos += 1
                if self._net_pos > self.score:
                    self.score = self._net_pos
                    reward += 1.0
                if nr < SCROLL_THRESHOLD:
                    self._world.insert(0, self._pop_next_row())
                    self._world = self._world[:VISIBLE_ROWS]
                    nr = SCROLL_THRESHOLD
            elif action == DOWN:
                self._net_pos -= 1

            nr = min(nr, VISIBLE_ROWS - 1)
            self._player_row = nr
            self._player_col = nc

            # --- 4. Collision check at new position -------------------------
            dest_row = self._world[self._player_row]
            if dest_row["type"] == ROAD:
                if self._car_at(self._player_row, self._player_col):
                    terminated = True
                    reward = -1.0
                    self._dead = True
            elif dest_row["type"] == WATER:
                if self._log_at(self._player_row, self._player_col) is None:
                    terminated = True
                    reward = -1.0
                    self._dead = True

        # --- 5. Auto-scroll -------------------------------------------------
        if not terminated:
            self._scroll_tick += 1
            interval = max(AUTOSCROLL_MIN,
                           int(AUTOSCROLL_START - (AUTOSCROLL_START - AUTOSCROLL_MIN)
                               * self._difficulty()))
            if self._scroll_tick >= interval:
                self._scroll_tick = 0
                self._world.insert(0, self._pop_next_row())
                self._world = self._world[:VISIBLE_ROWS]
                self._player_row += 1
                if self._player_row >= VISIBLE_ROWS:
                    self._player_row = VISIBLE_ROWS - 1
                    terminated = True
                    reward = -1.0
                    self._dead = True

        truncated = self._steps >= self.max_steps

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, {"score": self.score}

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        self._render_frame()

    def _init_pygame(self):
        if self._window is None:
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("Crossy Road — RL Env")
            self._clock = pygame.time.Clock()
            self._font  = pygame.font.SysFont("monospace", 18, bold=True)

    def _render_frame(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            self._init_pygame()

        canvas = pygame.Surface((SCREEN_W, SCREEN_H))

        for r, row in enumerate(self._world):
            y = r * TILE_SIZE

            if row["type"] == GRASS:
                canvas.fill(C_GRASS_A if r % 2 == 0 else C_GRASS_B,
                            (0, y, SCREEN_W, TILE_SIZE))

            elif row["type"] == ROAD:
                canvas.fill(C_ROAD, (0, y, SCREEN_W, TILE_SIZE))
                canvas.fill(C_KERB, (0, y,                  SCREEN_W, 4))
                canvas.fill(C_KERB, (0, y + TILE_SIZE - 4,  SCREEN_W, 4))
                for x in range(0, SCREEN_W, 20):
                    pygame.draw.rect(canvas, C_MARK,
                                     (x, y + TILE_SIZE // 2 - 1, 10, 2))
                for car in row["cars"]:
                    cx   = car["col"] * TILE_SIZE
                    body = C_CAR_FAST if car["speed"] == 1 else C_CAR_SLOW
                    pygame.draw.rect(canvas, body,
                                     (cx + 5, y + 14, TILE_SIZE - 10, TILE_SIZE - 26),
                                     border_radius=5)
                    pygame.draw.rect(canvas, C_WIN,
                                     (cx + 9, y + 18, TILE_SIZE - 18, 10),
                                     border_radius=3)
                    mid_y = y + 14
                    if car["dir"] == 1:
                        pts = [(cx + TILE_SIZE - 10, mid_y - 1),
                               (cx + TILE_SIZE - 5,  mid_y + 5),
                               (cx + TILE_SIZE - 15, mid_y + 5)]
                    else:
                        pts = [(cx + 10, mid_y - 1),
                               (cx + 5,  mid_y + 5),
                               (cx + 15, mid_y + 5)]
                    pygame.draw.polygon(canvas, C_MARK, pts)

            elif row["type"] == WATER:
                canvas.fill(C_WATER_A if r % 2 == 0 else C_WATER_B,
                            (0, y, SCREEN_W, TILE_SIZE))
                # Ripple lines
                for rx in range(8, SCREEN_W - 8, 28):
                    pygame.draw.arc(canvas, (60, 140, 220),
                                    (rx, y + TILE_SIZE // 2 - 4, 16, 8),
                                    0, 3.14, 2)
                for log in row["logs"]:
                    for i in range(log["length"]):
                        c  = (log["col"] + i) % GRID_COLS
                        lx = c * TILE_SIZE
                        # Log plank
                        pygame.draw.rect(canvas, C_LOG,
                                         (lx + 3, y + 16, TILE_SIZE - 6, TILE_SIZE - 32),
                                         border_radius=4)
                        # Ring detail on each plank
                        pygame.draw.rect(canvas, C_LOG_RING,
                                         (lx + 3, y + 16, TILE_SIZE - 6, TILE_SIZE - 32),
                                         width=2, border_radius=4)

        # Player (chicken)
        px = self._player_col * TILE_SIZE
        py = self._player_row * TILE_SIZE
        if self._dead:
            pygame.draw.line(canvas, (255, 0, 0),
                             (px + 14, py + 14), (px + TILE_SIZE - 14, py + TILE_SIZE - 14), 5)
            pygame.draw.line(canvas, (255, 0, 0),
                             (px + TILE_SIZE - 14, py + 14), (px + 14, py + TILE_SIZE - 14), 5)
        else:
            pygame.draw.rect(canvas, C_PLAYER,
                             (px + 14, py + 16, TILE_SIZE - 28, TILE_SIZE - 26),
                             border_radius=9)
            mid = px + TILE_SIZE // 2
            pygame.draw.polygon(canvas, C_BEAK,
                                [(mid - 5, py + 26), (mid + 5, py + 26), (mid, py + 35)])
            for ex in (px + 21, px + TILE_SIZE - 21):
                pygame.draw.circle(canvas, C_EYE,   (ex, py + 22), 5)
                pygame.draw.circle(canvas, C_SHINE, (ex - 1, py + 20), 2)

        # HUD
        if self._font:
            txt = self._font.render(f" Score: {self.score} ", True, C_HUD_TXT, C_HUD_BG)
            canvas.blit(txt, (4, 4))
            if self._dead:
                dead_txt = self._font.render(" DEAD — press any key ", True, C_DEAD_TXT, C_HUD_BG)
                canvas.blit(dead_txt, (SCREEN_W // 2 - dead_txt.get_width() // 2,
                                       SCREEN_H // 2 - 12))

        if self.render_mode == "human":
            self._window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), (1, 0, 2))

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
            self._window = None
