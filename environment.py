from typing import Optional
import numpy as np
import gymnasium as gym




class GridEnv(gym.Env):
    def __init__(self, length: int = 7, height: int = 6, obstacles: int = 3, render_mode = 'human'):
        # The length and height of the grid (7x6 by default)
        self.length = length
        self.height = height

        # The number of obstacles that the robot must avoid
        self.obstacles = obstacles

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._obstacle_locations = [np.array([-1, -1], dtype=np.int32) for _ in range(obstacles)]
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(np.array([0,0]), np.array([length - 1, height - 1]), shape=(2,), dtype=int),   
                "target": gym.spaces.Box(np.array([0,0]), np.array([length - 1, height - 1]), shape=(2,), dtype=int),  
            }
        )

        for i in range(obstacles):
            self.observation_space[f"obstacle_{i}"] = gym.spaces.Box(np.array([0,0]), np.array([length - 1, height - 1]), shape=(2,), dtype=int)
        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([1, 0]),   # Move right (positive x)
            1: np.array([0, 1]),   # Move up (positive y)
            2: np.array([-1, 0]),  # Move left (negative x)
            3: np.array([0, -1]),  # Move down (negative y)
        }
        
        self.render_mode = render_mode  


    def _get_obs(self):
        obs = {
            "agent": self._agent_location.copy(),
            "target": self._target_location.copy(),
        }
        # Always include every obstacle key
        for i in range(self.obstacles):
            obs[f"obstacle_{i}"] = self._obstacle_locations[i].copy()
        return obs
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._agent_location = self.np_random.integers(
            (0,0), (self.length - 1, self.height - 1), size=2, dtype=int
        )

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                (0,0), (self.length - 1, self.height - 1), size=2, dtype=int
            )

        # self._obstacle_locations = []
        for i in range(self.obstacles):
            while True:
                obstacle = np.array([
                    self.np_random.integers(0, self.length),
                    self.np_random.integers(0, self.height)
                ])

                # check collision with agent, target, or previous obstacles
                if (
                    not np.array_equal(obstacle, self._agent_location)
                    and not np.array_equal(obstacle, self._target_location)
                    and not any(np.array_equal(obstacle, o) for o in self._obstacle_locations)
                ):
                    self._obstacle_locations[i] = obstacle
                    break
        # print('Obstacles: ',self._obstacle_locations)

        observation = self._get_obs()
        info = self._get_info()

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

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        old_position = self._agent_location
        if action == 0 or action == 2:
            size = self.length
        elif action == 1 or action == 3:
            size = self.height

        self._agent_location = np.clip(
            self._agent_location + direction, 0, size - 1
        )

        print('Agent: ', self._agent_location)
        print('Obstacles: ', self._obstacle_locations)
        for obstacle in self._obstacle_locations:
            if np.array_equal(self._agent_location, obstacle):
                print('Old position')
                self._agent_location = old_position

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment for human viewing."""
        # print([o for o in self._obstacle_locations])
        if self.render_mode == "human":
            # Print a simple ASCII representation
            for y in range(self.height - 1, -1, -1):  # Top to bottom
                row = ""
                for x in range(self.length):
                    if np.array_equal([x, y], self._agent_location):
                        row += "A "  # Agent
                    elif np.array_equal([x, y], self._target_location):
                        row += "T "  # Target
                    elif any(np.array_equal([x, y], o) for o in self._obstacle_locations):
                        row += "O "  # Obstacle
                    else:
                        row += ". "  # Empty
                print(row)
            print()
    



learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1


