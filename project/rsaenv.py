import os
from typing import List, Tuple, Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from nwutil import Request, generate_sample_graph, LinkState


# Pre-defined paths between (source, destination) pairs.
# Each value is a list of exactly two paths; each path is a list of node IDs.
PATH_CANDIDATES: Dict[Tuple[int, int], List[List[int]]] = {
    (0, 3): [
        [0, 1, 2, 3],          # P1
        [0, 8, 7, 6, 3],       # P2
    ],
    (0, 4): [
        [0, 1, 5, 4],          # P3
        [0, 8, 7, 6, 3, 4],    # P4
    ],
    (7, 3): [
        [7, 1, 2, 3],          # P5
        [7, 6, 3],             # P6
    ],
    (7, 4): [
        [7, 1, 5, 4],          # P7
        [7, 6, 3, 4],          # P8
    ],
}


class RSAEnv(gym.Env):
    """
    Gymnasium environment for the Routing and Spectrum Allocation (RSA) problem.

    One episode = one request file (≈100 requests).

    At each time slot:
      1. Existing lightpaths age by one slot; expired ones are released.
      2. A new request (source, destination, holding_time) is current.
      3. Agent chooses action 0 or 1 = which pre-defined path to use.
      4. Env allocates the smallest-index wavelength that is free on all links
         of that path. If none found, the request is blocked.
      5. Reward: +1 for success, -1 for block.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        request_files: List[str],
        link_capacity: int = 20,
        max_holding_time: int = 100,
        shuffle_files: bool = True,
    ):
        """
        Args:
            request_files: List of paths to request files (train or eval).
                           One file is used per episode.
            link_capacity: Number of wavelengths per link (20 or 10).
            max_holding_time: Upper bound on holding_time values.
            shuffle_files: If True, pick a random file each reset;
                           if False, cycle through them.
        """
        super().__init__()

        assert len(request_files) > 0, "request_files list must not be empty"
        self.request_files = request_files
        self.link_capacity = int(link_capacity)
        self.max_holding_time = int(max_holding_time)
        self.shuffle_files = shuffle_files

        # Network topology
        self.graph = generate_sample_graph(capacity=self.link_capacity)
        self.edges: List[Tuple[int, int]] = sorted(self.graph.edges())
        self.num_edges = len(self.edges)
        self.num_nodes = self.graph.number_of_nodes()

        # Action: choose one of the two candidate paths
        self.action_space = spaces.Discrete(2)

        # Observation: link utilizations + current request info
        self.observation_space = spaces.Dict(
            {
                "link_utilizations": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_edges,),
                    dtype=np.float32,
                ),
                "source": spaces.Discrete(self.num_nodes),
                "destination": spaces.Discrete(self.num_nodes),
                "holding_time": spaces.Discrete(self.max_holding_time + 1),
            }
        )

        # Episode-specific state
        self.current_requests: List[Request] = []
        self.num_requests: int = 0
        self.request_index: int = 0  # index of current_request
        self.current_request: Request | None = None

        self.time_slot: int = 0
        self.blocked_count: int = 0
        self.total_count: int = 0

    # ----------------- Link state helpers ----------------- #

    def _get_link_state(self, u: int, v: int) -> LinkState:
        """Return LinkState object for undirected edge (u, v)."""
        if u > v:
            u, v = v, u
        return self.graph[u][v]["state"]

    def _reset_link_states(self):
        """Clear wavelength usage and utilization on all links."""
        for u, v in self.edges:
            state = self._get_link_state(u, v)
            if hasattr(state, "wavelength_states"):
                state.wavelength_states[:] = 0
            state.utilization = 0.0

    def _clock_forward(self):
        """
        Advance logical time by one slot:
        decrement holding time on all occupied wavelengths.
        """
        for u, v in self.edges:
            state = self._get_link_state(u, v)
            if hasattr(state, "wavelength_states"):
                busy = state.wavelength_states > 0
                state.wavelength_states[busy] -= 1
                state.utilization = float(
                    np.count_nonzero(state.wavelength_states)
                ) / state.capacity
        self.time_slot += 1

    def _find_available_wavelength(self, path: List[int]) -> int:
        """
        Find the smallest-index wavelength that is free on all links of 'path'.
        Returns wavelength index or -1 if none available.
        """
        for w in range(self.link_capacity):
            ok = True
            for u, v in zip(path[:-1], path[1:]):
                state = self._get_link_state(u, v)
                if state.wavelength_states[w] != 0:
                    ok = False
                    break
            if ok:
                return w
        return -1

    def _allocate_lightpath(self, path: List[int], wavelength: int, holding_time: int):
        """Allocate given wavelength along all links in 'path'."""
        for u, v in zip(path[:-1], path[1:]):
            state = self._get_link_state(u, v)
            assert state.wavelength_states[wavelength] == 0
            state.wavelength_states[wavelength] = holding_time
            state.utilization = float(
                np.count_nonzero(state.wavelength_states)
            ) / state.capacity

    def _get_link_utilizations(self) -> np.ndarray:
        """Return a vector of utilization values for all links."""
        utils = []
        for u, v in self.edges:
            state = self._get_link_state(u, v)
            utils.append(state.utilization)
        return np.asarray(utils, dtype=np.float32)

    # ----------------- Request handling ----------------- #

    def _choose_request_file(self) -> str:
        """Pick the next request file for this episode."""
        if self.shuffle_files:
            import random
            return random.choice(self.request_files)
        else:
            # deterministic cycling
            if not hasattr(self, "_file_index"):
                self._file_index = 0
            path = self.request_files[self._file_index]
            self._file_index = (self._file_index + 1) % len(self.request_files)
            return path

    def _load_requests_from_file(self, filepath: str) -> List[Request]:
        """
        Load requests from a file.

        Assumed format:
          - CSV or whitespace separated
          - Optional header line
          - Each data line: source, destination, holding_time
        """
        requests: List[Request] = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Skip header lines containing letters
                if any(c.isalpha() for c in line):
                    continue
                # Choose delimiter
                parts = line.split(",") if "," in line else line.split()
                if len(parts) < 3:
                    continue
                s, d, h = map(int, parts[:3])
                requests.append(Request(s, d, h))
        return requests

    # ----------------- Gymnasium API ----------------- #

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)

        req_file = self._choose_request_file()
        if not os.path.exists(req_file):
            raise FileNotFoundError(f"Request file not found: {req_file}")

        self.current_requests = self._load_requests_from_file(req_file)
        self.num_requests = len(self.current_requests)
        if self.num_requests == 0:
            raise ValueError(f"No requests loaded from file {req_file}")

        self.request_index = 0
        self.current_request = self.current_requests[0]

        # Reset network state and counters
        self._reset_link_states()
        self.time_slot = 0
        self.blocked_count = 0
        self.total_count = 0

        obs = self._get_observation()
        info = {"request_file": req_file}
        return obs, info

    def _get_observation(self):
        """Build the observation dict for the current state."""
        link_utils = self._get_link_utilizations()
        if self.current_request is None:
            src = dst = ht = 0
        else:
            src = self.current_request.source
            dst = self.current_request.destination
            ht = min(self.current_request.holding_time, self.max_holding_time)
        return {
            "link_utilizations": link_utils,
            "source": src,
            "destination": dst,
            "holding_time": ht,
        }

    def step(self, action: int):
        # 1. Age existing lightpaths
        self._clock_forward()

        assert self.current_request is not None, "No current request"
        req = self.current_request
        self.total_count += 1

        key = (req.source, req.destination)
        paths = PATH_CANDIDATES.get(key, None)

        blocked = False

        if paths is None:
            # No path defined for this pair
            blocked = True
        else:
            if not (0 <= action < len(paths)):
                raise ValueError(f"Invalid action {action} for request pair {key}")
            chosen_path = paths[action]

            # Smallest-index wavelength available on all links
            wavelength = self._find_available_wavelength(chosen_path)
            if wavelength == -1:
                blocked = True
            else:
                self._allocate_lightpath(chosen_path, wavelength, req.holding_time)

        # Reward
        if blocked:
            reward = -1.0
            self.blocked_count += 1
        else:
            reward = 1.0

        # Last request?
        is_last_request = self.request_index == (self.num_requests - 1)

        if is_last_request:
            terminated = True
            # obs after final step isn't really used, but keep shapes consistent
            next_obs = self._get_observation()
        else:
            self.request_index += 1
            self.current_request = self.current_requests[self.request_index]
            terminated = False
            next_obs = self._get_observation()

        truncated = False  # no separate time limit

        info = {
            "time_slot": self.time_slot,
            "blocked": blocked,
            "blocking_rate": self.blocked_count / max(self.total_count, 1),
        }

        return next_obs, reward, terminated, truncated, info

    def render(self):
        """Simple text render for debugging."""
        print(f"Time slot: {self.time_slot}")
        print(f"Current request: {self.current_request}")
        print(f"Blocking rate so far: {self.blocked_count}/{self.total_count}")
