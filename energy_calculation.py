# energy_calculation.py
import math

import numpy as np
from scipy.spatial import cKDTree
from utils.gpu_accelerator import GPUAccelerator
from utils.parallel_processor import ParallelProcessor


class EnergyCalculation:
    def __init__(self, sensorList, relayList):
        self.sensorList = np.array(sensorList)
        self.relayList = np.array(relayList)

    def energy_s(self):
        """Optimized sensor energy calculation"""
        bandwidth = 100.0
        E_radio_S = 50.0 * (10**(-9))
        Transmit_amplifier = 100.0 * (10**(-12))

        print("\n[⚙️ System] Starting energy calculations...")
        # Try GPU acceleration first
        energy_matrix = GPUAccelerator.calculate_energy_matrix(
            self.sensorList, self.relayList, bandwidth, E_radio_S, Transmit_amplifier)

        if energy_matrix is not None:
            print("[🚀 GPU] Successfully completed energy calculations using GPU")
            return energy_matrix.tolist()

        print(
            "[💻 CPU] GPU calculation failed or not available, using vectorized CPU for energy calculations")
        # Fall back to vectorized CPU processing

        # Vectorized distance calculation
        sensor_x = self.sensorList[:, 0].reshape(-1, 1)
        sensor_y = self.sensorList[:, 1].reshape(-1, 1)
        relay_x = self.relayList[:, 0].reshape(1, -1)
        relay_y = self.relayList[:, 1].reshape(1, -1)

        distances = np.sqrt(
            (sensor_x - relay_x)**2 +
            (sensor_y - relay_y)**2
        )

        # Vectorized energy calculation
        energy_matrix = bandwidth * \
            (E_radio_S + (Transmit_amplifier * (distances ** 2)))
        return energy_matrix.tolist()

    def energy_r(self):
        bandwidth_r = 200.0
        bandwidth_s = 100.0
        Energy_Relay_Relay = [[(0.0, 0.0, 0.0) for _ in range(len(self.relayList))]
                              for _ in range(len(self.relayList))]
        E_radio_R = 100.0 * (10 ** (-9))
        Transmit_amplifier = 100.0 * (10 ** (-12))
        E_aggregation = 0.00001
        for i in range(len(self.relayList)):
            for j in range(len(self.relayList)):
                dist = self.distance(abs(self.relayList[i][0] - self.relayList[j][0]),
                                     abs(self.relayList[i][1] - self.relayList[j][1]))
                energy_relay_tx = float(
                    bandwidth_r * (E_radio_R + (Transmit_amplifier * (dist ** 2))))
                energy_relay_rx = float(bandwidth_r * E_radio_R)
                energy_relay_rx_s = float(bandwidth_s * E_radio_R)
                total_energy = energy_relay_tx + energy_relay_rx
                Energy_Relay_Relay[i][j] = (
                    total_energy, energy_relay_rx_s, E_aggregation)
        return Energy_Relay_Relay

    def init_energy_r(self):
        """Initialize relay energy with higher values"""
        init_e_r = 1.0  # Significantly increased initial energy for relays
        nw_e_r = [[init_e_r for j in range(len(self.relayList))]
                  for i in range(len(self.relayList))]
        return nw_e_r

    def init_energy_s(self):
        """Initialize sensor energy with higher values"""
        init_e_s = 0.5  # Initial energy per sensor
        nw_e_s = [[init_e_s for j in range(len(self.relayList))]
                  for i in range(len(self.sensorList))]
        return nw_e_s

    @staticmethod
    def distance(a, b):
        return math.sqrt((a - b)**2)

    def calculate_load_balance(self, nw_e_s):
        """Enhanced load balance calculation with progressive thresholds"""
        total_energy = sum(sum(row) for row in nw_e_s)
        if total_energy == 0:
            return [1.0] * len(nw_e_s)

        avg_energy = total_energy / len(nw_e_s)
        load_factors = []

        for i in range(len(nw_e_s)):
            sensor_energy = sum(nw_e_s[i])
            neighbors = self._get_sensor_neighbors(i)

            if neighbors:
                # Calculate local and global balance factors
                neighbor_energies = [sum(nw_e_s[n]) for n in neighbors]
                local_avg = sum(neighbor_energies) / len(neighbors)

                # Progressive weighting based on energy level
                if sensor_energy > avg_energy:
                    # For high-energy nodes, focus more on global balance
                    global_weight = 0.8
                    local_weight = 0.2
                else:
                    # For low-energy nodes, focus more on local balance
                    global_weight = 0.2
                    local_weight = 0.8

                load_factor = (
                    global_weight * abs(sensor_energy - avg_energy) / (avg_energy + 1e-10) +
                    local_weight * abs(sensor_energy -
                                       local_avg) / (local_avg + 1e-10)
                )
            else:
                load_factor = abs(sensor_energy - avg_energy) / \
                    (avg_energy + 1e-10)

            load_factors.append(load_factor)

        return load_factors

    def _get_sensor_neighbors(self, sensor_idx, radius=None):
        if radius is None:
            radius = 1.0
        if not hasattr(self, '_kdtree'):
            self._kdtree = cKDTree(self.sensorList)
        sensor_pos = self.sensorList[sensor_idx]
        # Explicitly set p=2 so the parameter is filled.
        neighbors = self._kdtree.query_ball_point(sensor_pos, radius, p=2)
        return [i for i in neighbors if i != sensor_idx]

    def adjust_energy_distribution(self, nw_e_s, nw_e_r):
        """Adjust energy distribution for better load balancing"""
        load_factors = self.calculate_load_balance(nw_e_s)
        threshold = 0.2  # 20% threshold for load imbalance

        for i, load_factor in enumerate(load_factors):
            if load_factor > threshold:
                # Redistribute energy from heavily loaded sensors
                for j in range(len(nw_e_s[i])):
                    if nw_e_s[i][j] > 0:
                        nw_e_s[i][j] *= (1 - load_factor/2)

        return nw_e_s, nw_e_r


def compute_energy(energy_tuple):
    energy_value = energy_tuple[0]  # using first element; adjust if needed
    return energy_value
