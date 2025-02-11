# simulation.py
import csv
import os

import numpy as np


class Simulation:
    def __init__(self, sensorList, relayList, Fin_Conn_S_R, Fin_Conn_R_R, membership_values,
                 cluster_heads=None, rounds=10, result_dir="results"):
        # Ensure proper data types for coordinates
        self.sensorList = [(float(s[0]), float(s[1])) if isinstance(s, (list, tuple, np.ndarray))
                           else s for s in sensorList]
        self.relayList = [(float(r[0]), float(r[1])) for r in relayList]
        self.Fin_Conn_S_R = Fin_Conn_S_R
        self.Fin_Conn_R_R = Fin_Conn_R_R

        # Ensure membership_values is a list if provided as a NumPy array
        self.membership_values = membership_values
        if self.membership_values is not None and isinstance(self.membership_values, np.ndarray):
            self.membership_values = self.membership_values.tolist()

        # Ensure cluster_heads is a list if provided as a NumPy array
        self.cluster_heads = cluster_heads
        if self.cluster_heads is not None and isinstance(self.cluster_heads, np.ndarray):
            self.cluster_heads = self.cluster_heads.tolist()

        self.rounds = rounds
        self.init_energy = 0.0005 * len(sensorList) * len(relayList)
        self.nw_e_s = np.zeros(
            (len(sensorList), len(relayList)), dtype=np.float32)
        self.state_matrix_arr = np.zeros_like(
            self.nw_e_s, dtype=np.int8)  # Renamed from state_matrix
        self.result_dir = result_dir  # Store the config-specific directory

    def state_matrix(self):
        state_s = [[0 for j in range(len(self.relayList))]
                   for i in range(len(self.sensorList))]
        for i in range(len(self.sensorList)):
            for j in range(len(self.relayList)):
                if self.Fin_Conn_S_R[i][j] == 1:
                    state_s[i][j] = 1
        return state_s

    def init_s_counter(self, state_s):
        s_counter = [[0, 2] for i in range(len(state_s[0]))]
        for i in range(len(state_s[0])):
            for j in range(len(state_s)):
                if state_s[j][i] == 1:
                    s_counter[i][0] += 1
                    s_counter[i].append(j)
        return s_counter

    def init_sensor_ph_value(self):
        PH_list_s = [7.2 for j in range(len(self.sensorList))]
        return PH_list_s

    def master_ph_checker(self, PH_list_sensors):
        Clusters_with_oil_spills = []
        for i in range(len(self.Fin_Conn_S_R[0])):
            aggregated_PH = 0
            no_of_sensors = 0
            for j in range(len(self.Fin_Conn_S_R)):
                if self.Fin_Conn_S_R[j][i] == 1:
                    aggregated_PH += PH_list_sensors[j]
                    no_of_sensors += 1
            if no_of_sensors != 0:
                aggregated_PH /= no_of_sensors
                if aggregated_PH > 8.5:
                    Clusters_with_oil_spills.append(i)
        return Clusters_with_oil_spills

    def ph_checker(self, PH_list_sensors, cluster_head):
        aggregated_PH = 0
        no_of_sensors = 0
        for i in range(len(PH_list_sensors)):
            if self.Fin_Conn_S_R[i][cluster_head] == 1:
                aggregated_PH += PH_list_sensors[i]
                no_of_sensors += 1
        if no_of_sensors != 0:
            aggregated_PH /= no_of_sensors
            return aggregated_PH > 8.5
        else:
            return False

    def add_neighbour(self, Cluster_head, checked):
        neighbor_relays = []
        for i in range(len(self.Fin_Conn_R_R)):
            if self.Fin_Conn_R_R[i][Cluster_head] == 1 and i not in checked:
                neighbor_relays.append(i)
        return neighbor_relays

    def oil_simulator(self, PH_list_sensors):
        xrange = 50
        yrange = 50
        oil_spill_PH = 10
        for i in range(len(self.sensorList)):
            if self.sensorList[i][0] <= xrange and self.sensorList[i][1] <= yrange:
                PH_list_sensors[i] = oil_spill_PH

    def reset_oil(self, PH_list_sensors):
        normal_PH = 7.5
        for i in range(len(self.sensorList)):
            PH_list_sensors[i] = normal_PH

    def srp_toggler(self, state_s, s_counter, PH_list_sensors, round, srpfile):
        """Improved sleep scheduling with better energy awareness"""
        oil_affected_relays = self.master_ph_checker(PH_list_sensors)

        if not oil_affected_relays:
            for j in range(len(s_counter)):
                sensors_in_cluster = []
                scores = []

                for i in range(len(self.sensorList)):
                    if self.Fin_Conn_S_R[i][j] == 1:
                        sensors_in_cluster.append(i)
                        if self.membership_values is not None and j < len(self.membership_values):
                            # Include energy level in score calculation
                            membership_score = float(
                                self.membership_values[j][i])
                            energy_score = float(
                                sum(self.nw_e_s[i])) / self.init_energy
                            score = 0.5 * membership_score + 0.5 * energy_score
                        else:
                            score = 0.5
                        scores.append(score)

                if sensors_in_cluster:
                    # Dynamic sleep scheduling based on energy levels
                    total_energy = sum(sum(row) for row in self.nw_e_s)
                    energy_ratio = total_energy / \
                        (self.init_energy * len(self.sensorList))

                    # Adjust active ratio based on energy level
                    base_ratio = 0.3 if self.membership_values is not None else 0.5
                    active_ratio = min(base_ratio, max(0.2, energy_ratio))
                    active_count = max(
                        1, int(len(sensors_in_cluster) * active_ratio))

                    # Sort and activate top sensors
                    sorted_pairs = sorted(
                        zip(scores, sensors_in_cluster), reverse=True)
                    for i, (_, sensor_idx) in enumerate(sorted_pairs):
                        state_s[sensor_idx][j] = 1 if i < active_count else 0

                    srpfile.write(
                        f"Round {round}: Relay {j} - {active_count}/{len(sensors_in_cluster)} sensors active\n")
        else:
            # Oil spill detection - activate all sensors in affected areas
            srpfile.write(f"Oil Detected in Round: {round}\n")
            for i in oil_affected_relays:
                for j in range(len(state_s)):
                    if self.Fin_Conn_S_R[j][i] == 1:
                        state_s[j][i] = 1

    def save_to_csv(self, filename, headers, data):
        """Save CSV data ensuring we use config-specific directory"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(data)

    def simu_network(self, nw_e_s, nw_e_r, state_s):
        # Make deep copies of initial energy matrices
        init_energy_copy = [[e for e in row] for row in nw_e_s]
        self.nw_e_s = [[e for e in row] for row in nw_e_s]

        # Use subdirectory "network" under self.result_dir for network files
        network_path = os.path.join(self.result_dir, "network")
        network_file = open(os.path.join(
            network_path, "percentage_network.txt"), "w")
        srpfile = open(os.path.join(network_path, "srp_output.txt"), "w")
        network_file.write("sensors: " + str(len(self.sensorList)) + "\n")
        network_file.write("Relays: " + str(len(self.relayList)) + "\n")
        network_file.write("\n")
        total_energy = 0
        dead_s = 0
        dead_r = 0
        PH_list_sensors = self.init_sensor_ph_value()
        s_counter = self.init_s_counter(state_s)

        # Replace while-loop with for-loop
        for round in range(self.rounds):
            consumed_round_energy = 0.0
            network_file.write("ROUND: " + str(round) + "\n")
            srpfile.write("ROUND: " + str(round) + "\n")
            for i in range(len(self.Fin_Conn_S_R[0])):
                for j in range(len(self.Fin_Conn_S_R)):
                    if self.Fin_Conn_S_R[j][i] == 1 and state_s[j][i] == 1:
                        if nw_e_s[j][i] <= 0:
                            self.Fin_Conn_S_R[j][i] = 0
                            dead_s += 1
                        else:
                            # More efficient energy consumption for clustered nodes
                            base_energy = 0.01
                            try:
                                if self.membership_values is not None:
                                    cluster_idx = min(i % 3, 2)
                                    if j < len(self.membership_values[0]):
                                        membership_val = float(
                                            self.membership_values[cluster_idx][j])
                                        # Reduce energy consumption for high membership nodes
                                        energy_used = base_energy * \
                                            (0.8 + 0.2 * membership_val)
                                    else:
                                        energy_used = base_energy
                                else:
                                    energy_used = base_energy
                            except Exception as e:
                                print(f"Energy calculation warning: {e}")
                                energy_used = base_energy

                            nw_e_s[j][i] = max(0, nw_e_s[j][i] - energy_used)
                            consumed_round_energy += energy_used
                for w in range(len(self.Fin_Conn_R_R)):
                    if self.Fin_Conn_R_R[i][w] == 1:
                        # Reduce relay energy consumption
                        energy_used_r = 0.004  # Reduced from 0.005
                        if nw_e_r[i][w] > energy_used_r:
                            nw_e_r[i][w] -= energy_used_r
                            consumed_round_energy += energy_used_r
                        else:
                            self.Fin_Conn_R_R[i][w] = 0
                            dead_r += 1
                for k in range(len(self.Fin_Conn_R_R)):
                    if self.Fin_Conn_R_R[k][i] == 1:
                        if nw_e_r[k][i] <= 0:
                            self.Fin_Conn_R_R[k][i] = 0
                            for m in range(len(self.Fin_Conn_S_R)):
                                if self.Fin_Conn_S_R[m][i] != 0:
                                    self.Fin_Conn_S_R[m][i] = 0
                                    dead_s += 1
                            dead_r += 1
                        else:
                            energy_used_r = 0.005  # Use same fixed consumption
                            nw_e_r[k][i] -= energy_used_r
                            consumed_round_energy += energy_used_r
                            for l in range(len(self.Fin_Conn_S_R)):
                                if self.Fin_Conn_S_R[l][i] == 1 and state_s[l][i] == 1:
                                    if nw_e_r[k][i] > 0:
                                        nw_e_r[k][i] -= energy_used_r
                                        consumed_round_energy += energy_used_r
                                    else:
                                        self.Fin_Conn_R_R[k][i] = 0
                                        dead_r += 1
                                        for m in range(len(self.Fin_Conn_S_R)):
                                            if self.Fin_Conn_S_R[m][i] != 0:
                                                self.Fin_Conn_S_R[m][i] = 0
                                                dead_s += 1
                for m in range(len(self.Fin_Conn_R_R)):
                    if self.Fin_Conn_R_R[m][i] == 1:
                        energy_used_r = 0.005
                        nw_e_r[m][i] -= energy_used_r
                        consumed_round_energy += energy_used_r
            dead_s_pc = (dead_s / len(self.sensorList)) * 100
            dead_r_pc = (dead_r / len(self.relayList)) * 100
            dead_nw_pc = (dead_s_pc + dead_r_pc) / 2
            network_file.write("dead relays: " + str(dead_r) + "\n")
            network_file.write("Dead Network pc: " + str(dead_nw_pc) + " % \n")
            network_file.write("Dead Sensor pc: " + str(dead_s_pc) + " % \n")
            network_file.write("Dead relays pc: " + str(dead_r_pc) + " % \n")
            if round == 5:
                self.oil_simulator(PH_list_sensors)
            if round == 5 + len(self.relayList) - 1:
                self.reset_oil(PH_list_sensors)
            if dead_s_pc > 90 or dead_r_pc > 90:
                break
            self.srp_toggler(state_s, s_counter,
                             PH_list_sensors, round, srpfile)
            total_energy += consumed_round_energy
        network_file.write("total rounds: " + str(round+1) + "\n")
        network_file.write("total Energy used: " + str(total_energy) + "\n")
        network_file.close()
        srpfile.close()

        # Use self.result_dir
        base_dir = self.result_dir
        self.save_to_csv(os.path.join(base_dir, "energy", "final_energy_s.csv"), ["Sensor", "Energy"],
                         [[i, sum(row)] for i, row in enumerate(nw_e_s)])
        self.save_to_csv(os.path.join(base_dir, "energy", "final_energy_r.csv"), ["Relay", "Energy"],
                         [[i, sum(row)] for i, row in enumerate(nw_e_r)])
        self.save_to_csv(os.path.join(base_dir, "state", "state_s.csv"), ["Sensor", "State"],
                         [[i, sum(row)] for i, row in enumerate(state_s)])

        return nw_e_s, init_energy_copy  # <-- Added return tuple here
