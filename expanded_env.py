from typing import Optional
from collections import deque

import numpy as np
import gymnasium as gym

from stable_baselines3.common.callbacks import BaseCallback

class IntroduceBonusesCallback(BaseCallback):
    """
    Add bonus positions to the environment once the agent achieves a success threshold.
    """

    def __init__(self, success_rate, success_threshold=80, verbose=1):
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.success_rate = success_rate

    def done(self):
        env = self.training_env.envs[0].unwrapped
        return env.bonus == env.max_bonus

    def _on_step(self):

        # Check recent success rate
        env = self.training_env.envs[0].unwrapped

        if self.success_rate >= self.success_threshold - 10 * env.bonus:
            if env.bonus < env.max_bonus: 
                old_bonus = env.bonus
                env.bonus += 1
                self.success_rate = 0
                print(f"\nSuccess rate {self.success_rate:.2f} introducing {env.bonus} bonuses (was {old_bonus})!\n")
                # Reset to reinitialize bonus locations
                env.reset()
        return True


class GridEnv(gym.Env):
    def __init__(self, width: int = 7, height: int = 6, obstacles: int = 3, bonus: int = 3, render_mode = 'human'):
        # The width and height of the grid 
        self.height = height
        self.width = width

        # The number of obstacles that the robot must avoid
        self.obstacles = obstacles

        # The number of bonus positions that the robot must visit before the target
        # self.max_bonus = bonus
        self.bonus = bonus 
        self.hit_bonus = 0
        
        # A boolean that keeps the target position the agent must reach first locked when calculating distance
        self.locked = False
        self.keep_layout = False

        # Number of steps made
        self.steps = 0

        # Initialize positions, will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._obstacle_locations = [np.array([-1, -1], dtype=np.int32) for _ in range(obstacles)]
        self._bonus_locations = [np.array([-1, -1], dtype=np.int32) for _ in range(bonus)]
        self._final_location = np.array([-1, -1], dtype=np.int32)
        self._targets_location = [np.array([-1, -1], dtype=np.int32) for _ in range(bonus+1)]

        self.visited = set()

        # Define observation space 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "agent": gym.spaces.Box(np.array([0,0]), np.array([width - 1, height - 1]), shape=(2,), dtype=int),   
        #         "target": gym.spaces.Box(np.array([0,0]), np.array([width - 1, height - 1]), shape=(2,), dtype=int),  
        #     }
        # )

        # for i in range(obstacles):
        #     self.observation_space[f"obstacle_{i}"] = gym.spaces.Box(np.array([0,0]), np.array([width - 1, height - 1]), shape=(2,), dtype=int)
        
        # for i in range(bonus):
        #     self.observation_space[f"bonus_{i}"] = gym.spaces.Box(np.array([0,0]), np.array([width - 1, height - 1]), shape=(2,), dtype=int)
        
        # Define available actions (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),   # Move right (positive x)
            1: np.array([0, 1]),   # Move up (positive y)
            2: np.array([-1, 0]),  # Move left (negative x)
            3: np.array([0, -1]),  # Move down (negative y)
        }
        
        self.render_mode = render_mode  

    def pathfinder(self, items):
        """
        Returns the steps to the nearest target (bonus first, else final target)
        in [up, down, left, right] format, using BFS.
        """
        ax, ay = self._agent_location
        width, height = self.width, self.height

        obs_set = set(map(tuple, self._obstacle_locations))        
        items_set = set(map(tuple, items))

        # BFS: keep track of parent for path reconstruction
        q = deque()
        q.append((ax, ay))
        visited = set()
        visited.add((ax, ay))
        parent = {}  # child -> parent

        found_goal = None

        while q:
            x, y = q.popleft()
            if (x, y) in items_set:
                found_goal = (x, y)
                break

            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:  # down, up, right, left
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited and \
                    ((nx, ny) not in obs_set or obs_set == items_set):
                    visited.add((nx, ny))
                    q.append((nx, ny))
                    parent[(nx, ny)] = (x, y)

        # If no goal reachable, return zeros
        if found_goal is None:
            return np.zeros(4, dtype=np.float32)

        # Reconstruct path
        path = []
        node = found_goal
        while node != (ax, ay):
            path.append(node)
            node = parent[node]
        path.append((ax, ay))
        path = path[::-1]  # from agent -> goal

        # Count directional steps
        up = down = left = right = 0
        for i in range(1, len(path)):
            x0, y0 = path[i-1]
            x1, y1 = path[i]
            if y1 < y0:
                up += 1
            elif y1 > y0:
                down += 1
            elif x1 < x0:
                left += 1
            elif x1 > x0:
                right += 1

        return np.array([up, down, left, right], dtype=np.float32)


    def path_exists(self, start, goal):
        """
        Determines if a path exists from start to goal using BFS.
        
        Returns:
            bool: True if path exists, False otherwise
        """
        ax, ay = start
        gx, gy = goal
        width, height = self.width, self.height
        obs_set = set(map(tuple, self._obstacle_locations))

        q = deque()
        q.append((ax, ay))
        visited = set()
        visited.add((ax, ay))

        while q:
            x, y = q.popleft()
            if (x, y) == (gx, gy):
                return True

            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:  # down, up, right, left
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited and (nx, ny) not in obs_set:
                    visited.add((nx, ny))
                    q.append((nx, ny))

        return False
        
    def _get_obs(self):
        # distance to walls
        ax, ay = self._agent_location

        wall_up    = ay
        wall_down  = self.height - 1 - ay
        wall_left  = ax
        wall_right = self.width - 1 - ax
        wall_dist = np.array([wall_up, wall_down, wall_left, wall_right], dtype=np.float32)

        # distance to nearest obstacle (BFS)
        obs_dist = self.pathfinder(self._obstacle_locations)

        # distance to nearest target (BFS) 
        unvisited_bonuses = [b for b in self._bonus_locations if tuple(b) not in self.visited]

        if len(unvisited_bonuses) == 0:
        # All bonuses visited, go to final target
            target_dist = self.pathfinder([self._target_location])
        else:
        # Go to nearest unvisited bonus
            target_dist = self.pathfinder(unvisited_bonuses)

        
        obs = np.concatenate([
            wall_dist,
            obs_dist,
            target_dist,        
        ])

        return obs

    
    def _get_info(self):
        """Computes information about environment.

        Returns:
            dict: Info with distance between agent and target, and episode events
        """

        if self.locked:
            min_distance = np.linalg.norm(self._agent_location - self.target, ord=1)
        else:
            if self.hit_bonus == self.bonus:
                min_distance = np.linalg.norm(self._agent_location - self._target_location, ord=1)
                self.target = self._target_location
                self.final = True
                self.locked = True
            else:
                min_distance = self.width+self.height
                for bonus in self._bonus_locations:
                    if not(tuple(bonus) in self.visited):
                        distance = np.linalg.norm(bonus - self._target_location, ord=1)
                        if distance < min_distance:
                            min_distance = distance
                            self.target = bonus
                self.locked = True
                self.final = False

   
        return {
            "distance": min_distance,
            "timelimit": False,
            "hit_wall": False,
            "hit_obstacle": False,
            "hit_target":  False,
            "hit_bonus": self.hit_bonus
        }

    def reset(self, seed: Optional[int] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes

        Returns:
            tuple: (observation, info) for the initial state
        """
        
        if not(self.keep_layout): 
            super().reset(seed=seed)

            self.visited = set()
            self.hit_bonus = 0

            while True:
                # Randomly place the agent anywhere on the grid
                self._agent_location = self.np_random.integers(
                    (0,0), (self.width - 1, self.height - 1), size=2, dtype=int
                )

                # Randomly place target, ensuring it's different from agent position
                self._target_location = self._agent_location
                while np.array_equal(self._target_location, self._agent_location):
                    self._target_location = self.np_random.integers(
                        (0,0), (self.width - 1, self.height - 1), size=2, dtype=int
                    )
                
                # Randomly place obstacles
                for i in range(self.obstacles):
                    while True:
                        obstacle = np.array([
                            self.np_random.integers(0, self.width),
                            self.np_random.integers(0, self.height)
                        ])
                        # check collision with agent, target, and previous obstacles
                        if (
                            not np.array_equal(obstacle, self._agent_location)
                            and not np.array_equal(obstacle, self._target_location)
                            and not any(np.array_equal(obstacle, o) for o in self._obstacle_locations)
                        ):
                            self._obstacle_locations[i] = obstacle
                            break
                
                # Randomly place bonus locations
                if self.bonus > 0:
                    for i in range(self.bonus):
                        while True:
                            bonus = np.array([
                                self.np_random.integers(0, self.width),
                                self.np_random.integers(0, self.height)
                            ])
                            # check collision with agent, target, obstacles and previous bonuses
                            if (
                                not np.array_equal(bonus, self._agent_location)
                                and not np.array_equal(bonus, self._target_location)
                                and not any(np.array_equal(bonus, o) for o in self._obstacle_locations)
                                and not any(np.array_equal(bonus, b) for b in self._bonus_locations)
                            ):
                                self._bonus_locations[i] = bonus
                                break

                # Check if episode is solvable
                solvable = self.path_exists(self._agent_location, self._target_location)
                for i, bonus in enumerate(self._bonus_locations):
                    if i + 1 > self.bonus:
                        break
                    solvable = solvable and self.path_exists(self._agent_location, bonus)
                if solvable:
                    break
                # else:
                #     self.render()

            self.steps = 0
        
        observation = self._get_obs()
        info = self._get_info()
        self.old_distance = info['distance']

        return observation, info


    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        direction = self._action_to_direction[action]
        # Initialize reward, truncation and termination
        reward = 0
        truncated = False
        terminated = False
        
        # Attempt move and clip within bounds
        new_position = self._agent_location + direction
        clipped_position = np.clip(
            new_position,
            [0, 0],
            [self.width - 1, self.height - 1]
        )

        # Update agent position and distance
        self._agent_location = clipped_position
        info = self._get_info()
        observation = self._get_obs()

        # Check if agent hit any grid bounds
        if not np.array_equal(new_position, clipped_position):
            info['hit_wall'] = True
            terminated = True

        # Check if agent hit any of the obstacles
        for obstacle in self._obstacle_locations:
            if np.array_equal(self._agent_location, obstacle):
                info['hit_obstacle'] = True
                terminated = True

        self.keep_layout = False

        # Check if agent reached any bonus locations
        for bonus in self._bonus_locations:
            if np.array_equal(self._agent_location, bonus) and not(tuple(bonus) in self.visited):
                self.locked = False
                self.hit_bonus += 1
                reward += self.hit_bonus
                self.keep_layout = True

        # The episode ends due to time limit after a number of steps, based on the grid size
        self.steps += 1
        if self.steps > ((self.width) * (self.height)):
            info['timelimit'] = True
            truncated = True

        # Reward for exploring
        # if not(tuple(self._agent_location) in self.visited):
        #     reward += 0.05
        # else:
        #     reward -= 0.05

        # Reward for moving in the right direction
        reward += 0.5*(self.old_distance - info['distance']) #* (self.hit_bonus + 1)

        # Check if agent reached final target
        if np.array_equal(self._agent_location, self._target_location):
            if self.hit_bonus == self.bonus:
                reward += self.hit_bonus + 1
                info['hit_target'] = True
                terminated = True


        # Update distance, visited positions and determine if target should change
        self.old_distance = info['distance']
        self.visited.add(tuple(self._agent_location))
        self.locked = self.locked and not(terminated or truncated)

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment for human viewing."""

        if self.render_mode == "human":
            # Print a simple ASCII representation
            for y in range(self.height):# - 1, -1, -1):  # Top to bottom
                row = ""
                for x in range(self.width):
                    if np.array_equal([x, y], self._agent_location):
                        row += "A "  # Agent
                    elif np.array_equal([x, y], self._target_location):
                        row += "T "  # Target
                    elif any(np.array_equal([x, y], o) for o in self._obstacle_locations):
                        row += "O "  # Obstacle
                    elif any(np.array_equal([x, y], b) for b in self._bonus_locations):
                        row += "B "  # Bonus Locations
                    else:
                        row += ". "  # Empty
                print(row)
            print()
    