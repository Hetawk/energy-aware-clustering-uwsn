from numba import jit
import numba
from energy_calculation import EnergyCalculation  # Add this import
import skfuzzy as fuzz
from networkx.algorithms.approximation.steinertree import steiner_tree
from pulp import LpInteger, LpProblem, LpVariable, LpMinimize, PULP_CBC_CMD, lpSum
import networkx as nx
import logging
import numpy as np
import time
import math
import random
import warnings
from numba import NumbaPendingDeprecationWarning
import multiprocessing as mp  # Added for parallel solver threads
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

# optimization.py


@jit(nopython=True)
def calculate_distances(sensors, relays):
    """Vectorized distance calculation"""
    sensors = np.array(sensors)
    relays = np.array(relays)
    distances = np.zeros((len(sensors), len(relays)))

    for i in range(len(sensors)):
        distances[i] = np.sqrt(
            (sensors[i, 0] - relays[:, 0])**2 +
            (sensors[i, 1] - relays[:, 1])**2
        )
    return distances


class Optimization:
    def __init__(self, rad, sen, width, relay_constraint, clustering):
        self.width = float(width)  # Width of the network 100.0
        self.R = float(rad)
        self.relayConstraint = relay_constraint  # Number of relays 121
        self.sensorList = []
        self.relayList = []
        self.clustering = clustering
        self.membership_values = None  # Initialize membership_values to None
        self.cluster_heads = None  # Initialize cluster_heads to None
        self.populate_sensor_relay_lists(sen)
        if clustering:
            self.apply_clustering()
            self.cluster_heads = self.select_cluster_heads()
        if self.membership_values is not None and isinstance(self.membership_values, np.ndarray):
            self.membership_values = self.membership_values.tolist()
        if self.cluster_heads is not None and isinstance(self.cluster_heads, np.ndarray):
            self.cluster_heads = self.cluster_heads.tolist()

    def populate_sensor_relay_lists(self, sen):
        for i in range(sen):
            self.sensorList.append((round(random.uniform(0.0, self.width), 6), round(
                random.uniform(0.0, self.width), 6)))
        row = 0
        col = 0
        while row <= self.width:
            while col <= self.width:
                self.relayList.append((float(row), float(col)))
                col += 10
            row += 10
            col = 0

    def distance(self, a, b):
        return math.sqrt((a ** 2) + (b ** 2))

    def connection_s_r(self):
        # Replace loop-based distance calculation with vectorized version
        distances = calculate_distances(self.sensorList, self.relayList)
        return (distances <= self.R).astype(int)

    def connection_r_r(self):
        Neighbor_Relay_Relay = [
            [0 for i in range(len(self.relayList))] for j in range(len(self.relayList))
        ]
        for i in range(len(self.relayList)):
            for j in range(len(self.relayList)):
                dist = self.distance(abs(self.relayList[i][0] - self.relayList[j][0]),
                                     abs(self.relayList[i][1] - self.relayList[j][1]))
                if i == j:
                    Neighbor_Relay_Relay[i][j] = 0
                elif dist <= self.R:
                    Neighbor_Relay_Relay[i][j] = 1
                else:
                    Neighbor_Relay_Relay[i][j] = 0
        return Neighbor_Relay_Relay

    def link_s_r(self):
        bandwidth = 100.0
        Linkflow_Sensor_Relay = [
            [0 for i in range(len(self.relayList))] for j in range(len(self.sensorList))
        ]
        for i in range(len(self.sensorList)):
            for j in range(len(self.relayList)):
                Linkflow_Sensor_Relay[i][j] = bandwidth
        return Linkflow_Sensor_Relay

    def link_r_r(self):
        bandwidth = 200.0
        Linkflow_Relay_Relay = [
            [0 for i in range(len(self.relayList))] for j in range(len(self.relayList))
        ]
        for i in range(len(self.relayList)):
            for j in range(len(self.relayList)):
                if i != j:
                    Linkflow_Relay_Relay[i][j] = bandwidth
                else:
                    Linkflow_Relay_Relay[i][j] = 0
        return Linkflow_Relay_Relay

    def energy_s_r(self):
        bandwidth = 100.0
        Energy_Sensor_Relay = [
            [0 for i in range(len(self.relayList))] for j in range(len(self.sensorList))
        ]
        E_radio_S = 50.0 * (10 ** (-9))
        E_radio_R = 100.0 * (10 ** (-9))
        Transmit_amplifier = 100.0 * (10 ** (-12))
        for i in range(len(self.sensorList)):
            for j in range(len(self.relayList)):
                dist = self.distance(abs(self.sensorList[i][0] - self.relayList[j][0]),
                                     abs(self.sensorList[i][1] - self.relayList[j][1]))
                energy_sensor_tx = float(
                    bandwidth *
                    (E_radio_S + (Transmit_amplifier * (dist ** 2)))
                )
                energy_relay_rx = float(bandwidth * E_radio_R)
                total_energy = energy_sensor_tx + energy_relay_rx
                Energy_Sensor_Relay[i][j] = total_energy
        return Energy_Sensor_Relay

    def energy_r_r(self):
        bandwidth = 200.0
        Energy_Relay_Relay = [
            [0 for i in range(len(self.relayList))] for j in range(len(self.relayList))
        ]
        E_radio_R = 100.0 * (10 ** (-9))
        Transmit_amplifier = 100.0 * (10 ** (-12))
        for i in range(len(self.relayList)):
            for j in range(len(self.relayList)):
                dist = self.distance(abs(self.relayList[i][0] - self.relayList[j][0]),
                                     abs(self.relayList[i][1] - self.relayList[j][1]))
                energy_relay_tx = float(
                    bandwidth *
                    (E_radio_R + (Transmit_amplifier * (dist ** 2)))
                )
                energy_relay_rx = float(bandwidth * E_radio_R)
                total_energy = energy_relay_tx + energy_relay_rx
                Energy_Relay_Relay[i][j] = total_energy
        return Energy_Relay_Relay

    def apply_clustering(self):
        """Apply Fuzzy C-Means clustering with energy consideration"""
        try:
            sensor_data = np.array(self.sensorList)
            sensor_data = sensor_data[:, :2]  # Only use x,y coordinates
            n_clusters = 3   # Fuzzy clusters' number is set to 3
            error = 0.005
            maxiter = 1000
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                sensor_data.T, n_clusters, 2, error=error,
                maxiter=maxiter, init=None
            )
            self.cluster_labels = np.argmax(u, axis=0)
            self.membership_values = u
            # ...existing sensor augmentation with cluster info...
        except Exception as e:
            print(f"Warning: Clustering error: {e}")
            self.membership_values = None
            self.cluster_labels = np.zeros(len(self.sensorList))

    def select_cluster_heads(self):
        """Improved cluster head selection with better energy distribution"""
        try:
            cluster_heads = []
            energy_calc = EnergyCalculation(self.sensorList, self.relayList)
            sensor_energies = energy_calc.init_energy_s()
            total_energy = sum(sum(row) for row in sensor_energies)
            energy_ratios = [sum(row)/total_energy for row in sensor_energies]
            centroids = np.zeros((3, 2))
            for i in range(3):
                mask = self.cluster_labels == i
                if np.any(mask):
                    points = np.array(
                        [s[:2] for s, m in zip(self.sensorList, mask) if m])
                    centroids[i] = np.mean(points, axis=0)
                else:
                    centroids[i] = np.array([0.0, 0.0])
            for cluster in range(3):
                cluster_nodes = []
                scores = []
                for i, sensor in enumerate(self.sensorList):
                    if isinstance(sensor, (list, np.ndarray)) and len(sensor) > 2:
                        if int(sensor[2]) == cluster:
                            energy_score = energy_ratios[i]
                            membership_score = self.membership_values[cluster][i]
                            sensor_pos = np.array(sensor[:2])
                            dist_to_centroid = np.linalg.norm(
                                sensor_pos - centroids[cluster])
                            max_dist = np.sqrt(2) * self.width
                            position_score = 1 - (dist_to_centroid / max_dist)
                            combined_score = (0.5 * energy_score +
                                              0.3 * membership_score +
                                              0.2 * position_score)
                            cluster_nodes.append(sensor)
                            scores.append(combined_score)
                if cluster_nodes:
                    max_score_idx = scores.index(max(scores))
                    cluster_heads.append(cluster_nodes[max_score_idx])
                    selected_idx = self.sensorList.index(
                        cluster_nodes[max_score_idx])
                    energy_ratios[selected_idx] *= 0.8
            return cluster_heads
        except Exception as e:
            print(f"Error in cluster head selection: {e}")
            return []

    def solve_problem(self):
        """Solve the optimization problem with timing and parallel processing"""
        start_time = time.time()
        try:
            Problem = LpProblem("The Energy Problem", LpMinimize)
            Var_S_R = [LpVariable(str('SR' + str(i) + "_" + str(j)), 0, 1, LpInteger)
                       for i in range(len(self.sensorList)) for j in range(len(self.relayList))]
            Var_R_R = [LpVariable(str('RR' + str(i) + "_" + str(j)), 0, 1, LpInteger)
                       for i in range(len(self.relayList)) for j in range(len(self.relayList))]
            Bool_Var = [LpVariable(str('B' + str(i)), 0, 1, LpInteger)
                        for i in range(len(self.relayList))]

            n_s_r = self.connection_s_r()
            n_r_r = self.connection_r_r()
            e_s_r = self.energy_s_r()
            e_r_r = self.energy_r_r()
            l_s_r = self.link_s_r()
            l_r_r = self.link_r_r()

            k = 0
            objective = []
            conn_SR = []
            conn_RR = []

            while k < len(Var_S_R):
                for i in range(len(e_s_r)):
                    for j in range(len(e_s_r[i])):
                        objective.append(e_s_r[i][j] * Var_S_R[k])
                        conn_SR.append(n_s_r[i][j] * Var_S_R[k])
                        k += 1
                    Problem += lpSum(conn_SR) == 1
                    conn_SR = []

            B_Max = 3000
            B_SR = 100
            Var_alias_SR = [str(i) for i in Var_S_R]
            booleansumcolumn = []
            linkflow_column = []

            for i in range(len(self.relayList)):
                for j in range(len(self.sensorList)):
                    e = Var_alias_SR.index(str('SR' + str(j) + "_" + str(i)))
                    booleansumcolumn.append(Var_S_R[e])
                    linkflow_column.append(l_s_r[j][i] * Var_S_R[e])
                Problem += lpSum(booleansumcolumn) >= Bool_Var[i]
                Problem += lpSum(linkflow_column) <= B_Max
                for c in booleansumcolumn:
                    Problem += c <= Bool_Var[i]
                booleansumcolumn = []
                linkflow_column = []

            k = 0
            while k < len(Var_R_R):
                for i in range(len(e_r_r)):
                    for j in range(len(e_r_r[i])):
                        objective.append(e_r_r[i][j] * Var_R_R[k])
                        conn_RR.append(n_r_r[i][j] * Var_R_R[k])
                        k += 1
                    Problem += lpSum(conn_RR) >= 0
                    conn_RR = []

            Var_alias_RR = [str(i) for i in Var_R_R]
            booleansumcolumn = []

            for i in range(len(self.relayList)):
                for j in range(len(self.relayList)):
                    e = Var_alias_RR.index(str('RR' + str(j) + "_" + str(i)))
                    booleansumcolumn.append(Var_R_R[e])
                Problem += lpSum(booleansumcolumn) >= Bool_Var[i]
                for c in booleansumcolumn:
                    Problem += c <= Bool_Var[i]
                booleansumcolumn = []

            for i in range(len(self.relayList)):
                for j in range(len(self.relayList)):
                    if i != j:
                        e = Var_alias_RR.index(
                            str('RR' + str(j) + "_" + str(i)))
                        f = Var_alias_RR.index(
                            str('RR' + str(i) + "_" + str(j)))
                        Problem += Var_R_R[e] == Var_R_R[f]
                    else:
                        f = Var_alias_RR.index(
                            str('RR' + str(i) + "_" + str(j)))
                        Problem += Var_R_R[f] == 0

            Problem += lpSum(objective)
            Problem += lpSum(Bool_Var) <= self.relayConstraint

            t1 = time.perf_counter()
            logging.info(
                f"[Optimization] Starting solver with gapRel=1e-13 and threads={mp.cpu_count()}")
            Problem.solve(PULP_CBC_CMD(gapRel=1e-13, threads=mp.cpu_count()))
            t2 = time.perf_counter()
            optimization_duration = t2 - t1   # Calculate solver duration
            logging.info(
                f"[Optimization] Solved in {optimization_duration:.2f}s")
            return Problem, optimization_duration
        except Exception as e:
            print(f"[❌ Error] Optimization failed: {str(e)}")
            return None, time.time() - start_time
