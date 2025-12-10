import networkx as nx
import numpy as np


class Request:
    def __init__(self, source: int, destination: int, holding_time: int):
        self.source = source
        self.destination = destination
        self.holding_time = holding_time

    def __repr__(self):
        return f"Request(src={self.source}, dst={self.destination}, hold={self.holding_time})"


class BaseLinkState:
    def __init__(self, u, v, capacity=20, utilization=0.0):
        # store endpoints in sorted order so (u,v) and (v,u) are the same
        if u > v:
            u, v = v, u
        self.endpoints = (u, v)
        self.capacity = capacity
        self.utilization = utilization

    def __repr__(self):
        return f"LinkState(capacity={self.capacity}, util={self.utilization})"


class LinkState(BaseLinkState):
    """
    Data structure to store the link state.

    This extends BaseLinkState with a per-wavelength occupancy array:
      - wavelength_states[i] == 0  -> wavelength i is free
      - wavelength_states[i]  > 0  -> remaining holding time for wavelength i
    """
    def __init__(self, u, v, capacity=20, utilization=0.0):
        super().__init__(u, v, capacity, utilization)
        # remaining holding time on each wavelength (0 means free)
        self.wavelength_states = np.zeros(self.capacity, dtype=int)

    def reset(self):
        """Reset this link to an empty state."""
        self.wavelength_states.fill(0)
        self.utilization = 0.0

    def step_time(self):
        """
        Advance time by one slot on this link:
        decrement remaining holding times and free expired wavelengths.
        """
        busy = self.wavelength_states > 0
        self.wavelength_states[busy] -= 1
        self._update_utilization()

    def occupy(self, wavelength: int, holding_time: int):
        """Mark a wavelength as occupied for 'holding_time' future time slots."""
        assert 0 <= wavelength < self.capacity
        assert self.wavelength_states[wavelength] == 0
        self.wavelength_states[wavelength] = int(holding_time)
        self._update_utilization()

    def _update_utilization(self):
        """Update utilization ratio based on number of busy wavelengths."""
        busy_count = int(np.count_nonzero(self.wavelength_states))
        self.utilization = busy_count / float(self.capacity)


def generate_sample_graph(capacity: int = 20):
    """
    Create the sample 9-node topology used in the RSA project.

    Args:
        capacity: number of wavelength slots on every physical link.
    """
    G = nx.Graph()

    G.add_nodes_from(range(9))

    # Define links: ring links + extra links
    links = [(n, (n + 1) % 9) for n in range(9)] + [(1, 7), (1, 5), (3, 6)]

    # Add edges with link state objects
    for u, v in links:
        G.add_edge(u, v, state=LinkState(u, v, capacity=capacity))
    return G
