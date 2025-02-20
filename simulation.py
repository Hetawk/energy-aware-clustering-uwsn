# simulation.py
import csv
import os
import concurrent.futures
import signal
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import logging
import time

import numpy as np

from energy_calculation import EnergyCalculation
from optimization import Optimization

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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

    @staticmethod
    def init_s_counter(state_s):
        s_counter = [[0, 2] for _ in range(len(state_s[0]))]
        for col in range(len(state_s[0])):
            for row in range(len(state_s)):
                if state_s[row][col] == 1:
                    s_counter[col][0] += 1
                    s_counter[col].append(row)
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

    def srp_toggler(self, state_s, s_counter, PH_list_sensors, rnd, srpfile):
        """Improved sleep scheduling with better energy awareness"""
        oil_affected_relays = self.master_ph_checker(PH_list_sensors)

        if not oil_affected_relays:
            for j in range(len(s_counter)):
                sensors_in_cluster = []
                scores = []
                energy_levels = []
                for i in range(len(self.sensorList)):
                    if self.Fin_Conn_S_R[i][j] == 1:
                        sensors_in_cluster.append(i)
                        # If clustering is enabled, use the fuzzy membership for relay j
                        if self.membership_values is not None and j < len(self.membership_values):
                            # Instead of sensor[2], use self.cluster_labels to check membership
                            membership_score = float(
                                self.membership_values[j][i])
                            energy_score = float(
                                sum(self.nw_e_s[i])) / self.init_energy
                            score = 0.5 * membership_score + 0.5 * energy_score
                        else:
                            score = 0.5
                        scores.append(score)
                        energy_levels.append(
                            float(sum(self.nw_e_s[i])) / self.init_energy)
                if sensors_in_cluster:
                    # Adjust active count based on average remaining energy in the cluster
                    avg_energy = np.mean(energy_levels) if energy_levels else 0
                    # Dynamic ratio between 0.25 and 0.5 based on energy status
                    active_ratio = 0.25 + (0.25 * avg_energy)
                    active_count = max(
                        1, int(len(sensors_in_cluster) * active_ratio))
                    sorted_pairs = sorted(
                        zip(scores, sensors_in_cluster), reverse=True)
                    for idx, (_, sensor_idx) in enumerate(sorted_pairs):
                        state_s[sensor_idx][j] = 1 if idx < active_count else 0
                    srpfile.write(
                        f"Round {rnd}: Relay {j} - {active_count}/{len(sensors_in_cluster)} sensors active\n")
        else:
            srpfile.write(f"Oil Detected in Round: {rnd}\n")
            for i in oil_affected_relays:
                for j in range(len(state_s)):
                    if self.Fin_Conn_S_R[j][i] == 1:
                        state_s[j][i] = 1

    @staticmethod
    def save_to_csv(filename, headers, data):
        """Save CSV data ensuring we use config-specific directory"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(data)

    def simu_network(self, nw_e_s):
        simulation_start_time = time.time()
        init_energy_copy = np.copy(nw_e_s)
        self.nw_e_s = np.copy(nw_e_s)
        network_path = os.path.join(self.result_dir, "network")
        os.makedirs(network_path, exist_ok=True)
        try:
            with open(os.path.join(network_path, "percentage_network.txt"), "w") as network_file:
                network_file.write(
                    f"sensors: {len(self.sensorList)}\nRelays: {len(self.relayList)}\n\n")
                total_energy = 0
                # Removed unused variables: PH_list_sensors, s_counter and srpfile.
                for rnd in range(self.rounds):
                    # Dummy simulation logic; implement your simulation here.
                    pass
                simulation_duration = time.time() - simulation_start_time
                network_file.write(
                    f"Simulation Duration (s): {simulation_duration}\n")
                network_file.write(f"total rounds: {rnd+1}\n")
                network_file.write(f"total Energy used: {total_energy}\n")
            return nw_e_s, init_energy_copy
        except Exception as e:
            logging.error(f"Simulation failed: {str(e)}", exc_info=True)
            return None, None


def run_single_simulation(radius, count, rounds, width, relay_constraint, clustering, result_dir):
    optimization = Optimization(
        radius, count, width, relay_constraint, clustering=clustering)
    Problem, time_taken = optimization.solve_problem()

    # Initialize energy calculation
    energy_calculation = EnergyCalculation(
        optimization.sensorList, optimization.relayList)
    nw_e_s = energy_calculation.init_energy_s()

    # Create simulation instance
    sim_instance = Simulation(
        optimization.sensorList,
        optimization.relayList,
        optimization.connection_s_r(),
        optimization.connection_r_r(),
        optimization.membership_values,
        optimization.cluster_heads,
        rounds,
        result_dir=result_dir
    )

    # Removed unused assignment of state_s
    final_energy_s, init_energy_s = sim_instance.simu_network(nw_e_s)

    consumed = sum(sum(row) for row in init_energy_s) - \
        sum(sum(row) for row in final_energy_s)
    relays = len(optimization.relayList)
    energy = consumed

    return relays, energy, time_taken


def run_simulations(radius, sensor_counts, rounds, width, relay_constraint, clustering, result_dir, interrupt_handler=None):
    """Run simulations with interrupt handling"""
    relays = []
    energies = []
    times = []

    # Create directories
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "network"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "energy"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "state"), exist_ok=True)
    os.makedirs(os.path.join(
        result_dir, "evaluation", "metrics"), exist_ok=True)

    def sigint_handler(_, __):
        """Handle Ctrl+C during simulation"""
        print("\n\n[⚠️ Interrupt] Caught interrupt signal...")
        try:
            answer = input(
                '[⚠️ Interrupt] Do you want to stop the current run? (y/N): ').strip().lower()
            if answer == 'y':
                print("\n[🛑 Stopping] Gracefully stopping current execution...")
                # Signal all worker processes to stop
                for _, future in futures:
                    future.cancel()
                executor.shutdown(wait=False)
                return True
            print("\n[▶️ Continuing] Resuming execution...")
            return False
        except (EOFError, KeyboardInterrupt):
            print("\n[🛑 Stopping] Forced stop...")
            return True

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = []
            for count in sensor_counts:
                if interrupt_handler and interrupt_handler.should_stop:
                    print("[🛑 Stopping] Interrupt received")
                    break

                future_result = executor.submit(
                    run_single_simulation,
                    radius, count, rounds, width,
                    relay_constraint, clustering, result_dir
                )
                futures.append((count, future_result))

            # Process results as they complete
            for count, future_result in futures:
                if interrupt_handler and interrupt_handler.check():
                    print(
                        "[🛑 Stopping] Interrupt received, cancelling remaining tasks...")
                    for future_to_cancel in futures:
                        future_to_cancel[1].cancel()
                    break

                try:
                    result = future_result.result(
                        timeout=300)  # 5-minute timeout
                    if result:
                        relay, energy, elapsed_time = result
                        relays.append(relay)
                        energies.append(energy)
                        times.append(elapsed_time)
                        print(f"[✅ Complete] Processed {count} sensors")
                except concurrent.futures.TimeoutError:
                    print(
                        f"[❌ Error] Simulation timed out for {count} sensors")
                except Exception as e:
                    print(
                        f"[❌ Error] Failed processing {count} sensors: {str(e)}")

    except KeyboardInterrupt:
        print("\n[⚠️ Interrupt] Stopping simulations...")
        if interrupt_handler:
            public_signal_handler = interrupt_handler.get_signal_handler()
            public_signal_handler(signal.SIGINT, None)

    finally:
        # Save any completed results
        if relays:
            from utils.result_saver import save_partial_results
            completed_counts = sensor_counts[:len(relays)]
            save_partial_results(
                result_dir, completed_counts, relays, energies, times)
            print(
                f"[💾 Saved] Results for {len(completed_counts)} completed simulations")

    return relays, energies, times
