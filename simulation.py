# simulation.py
import csv
import os
import concurrent.futures
import signal
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import logging

import numpy as np

from energy_calculation import EnergyCalculation
from optimization import Optimization

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def simulate_round_wrapper(args):
    """Wrapper function for parallel execution of simulate_round"""
    self, round, nw_e_s, nw_e_r, state_s, network_file, srpfile, PH_list_sensors, s_counter = args
    return self.simulate_round(round, nw_e_s, nw_e_r, state_s, network_file, srpfile, PH_list_sensors, s_counter)

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

                    # Reduce output frequency
                    if round % 10 == 0:
                        srpfile.write(
                            f"Round {round}: Relay {j} - {active_count}/{len(sensors_in_cluster)} sensors active\n")
        else:
            # Oil spill detection - activate all sensors in affected areas
            # Reduce output frequency
            if round % 10 == 0:
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

    def simulate_round(self, round, nw_e_s, nw_e_r, state_s, network_file, srpfile, PH_list_sensors, s_counter):
        """Simulate a single round of the network"""
        consumed_round_energy = 0.0
        # Reduce output frequency
        if round % 10 == 0:
            network_file.write("ROUND: " + str(round) + "\n")
            srpfile.write("ROUND: " + str(round) + "\n")

        # Vectorized energy consumption calculation
        active_connections = (self.Fin_Conn_S_R == 1) & (state_s == 1)
        energy_used_s = np.where(
            active_connections, 0.01, 0)  # Base energy

        # Apply membership-based energy reduction
        if self.membership_values is not None:
            cluster_idx = np.minimum(
                np.arange(len(self.relayList)) % 3, 2)
            membership_vals = np.array([self.membership_values[cluster_idx[j]][i]
                                        if j < len(self.membership_values[0]) else 0
                                        for i in range(len(self.sensorList)) for j in range(len(self.relayList))]).reshape(len(self.sensorList), len(self.relayList))
            energy_used_s = np.where(
                active_connections, energy_used_s * (0.8 + 0.2 * membership_vals), 0)

        # Update sensor energies
        nw_e_s = np.maximum(0, nw_e_s - energy_used_s)
        consumed_round_energy += np.sum(energy_used_s)

        # Relay energy consumption
        energy_used_r = np.where(self.Fin_Conn_R_R == 1, 0.004, 0)
        nw_e_r = np.maximum(0, nw_e_r - energy_used_r)
        consumed_round_energy += np.sum(energy_used_r)

        # Update dead nodes
        dead_s = np.sum(nw_e_s <= 0)
        dead_r = np.sum(nw_e_r <= 0)

        dead_s_pc = (dead_s / len(self.sensorList)) * 100
        dead_r_pc = (dead_r / len(self.relayList)) * 100
        dead_nw_pc = (dead_s_pc + dead_r_pc) / 2
        # Reduce output frequency
        if round % 10 == 0:
            network_file.write("dead relays: " + str(dead_r) + "\n")
            network_file.write("Dead Network pc: " +
                               str(dead_nw_pc) + " % \n")
            network_file.write("Dead Sensor pc: " +
                               str(dead_s_pc) + " % \n")
            network_file.write("Dead relays pc: " +
                               str(dead_r_pc) + " % \n")
        if round == 5:
            self.oil_simulator(PH_list_sensors)
        if round == 5 + len(self.relayList) - 1:
            self.reset_oil(PH_list_sensors)
        if dead_s_pc > 90 or dead_r_pc > 90:
            return nw_e_s, nw_e_r, True, consumed_round_energy  # Indicate early termination

        self.srp_toggler(state_s, s_counter,
                         PH_list_sensors, round, srpfile)
        return nw_e_s, nw_e_r, False, consumed_round_energy

    def simu_network(self, nw_e_s, nw_e_r, state_s):
        """Run network simulation with proper directory handling"""
        # Make deep copies of initial energy matrices
        init_energy_copy = np.copy(nw_e_s)
        self.nw_e_s = np.copy(nw_e_s)

        # Ensure network directory exists
        network_path = os.path.join(self.result_dir, "network")
        os.makedirs(network_path, exist_ok=True)

        try:
            # Open output files
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

            # Prepare arguments for parallel execution
            round_args = [(self, round, np.copy(nw_e_s), np.copy(nw_e_r), np.copy(state_s), network_file, srpfile, PH_list_sensors, s_counter)
                          for round in range(self.rounds)]

            # Use ProcessPoolExecutor to parallelize the simulation loop
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                results = executor.map(simulate_round_wrapper, round_args)

                # Process results from each round
                for round, (new_nw_e_s, new_nw_e_r, early_terminate, consumed_round_energy) in zip(range(self.rounds), results):
                    nw_e_s = new_nw_e_s
                    nw_e_r = new_nw_e_r
                    total_energy += consumed_round_energy

                    if early_terminate:
                        print(f"Early termination at round {round}")
                        break

            network_file.write("total rounds: " + str(round+1) + "\n")
            network_file.write("total Energy used: " +
                               str(total_energy) + "\n")
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

            logging.info(
                f"Simulation completed successfully for {len(self.sensorList)} sensors.")
            return nw_e_s, init_energy_copy  # <-- Added return tuple here

        except Exception as e:
            logging.error(f"Simulation failed: {str(e)}", exc_info=True)
            return None, None


def run_single_simulation(radius, count, rounds, width, relay_constraint, clustering, result_dir):
    """Run a single simulation iteration"""
    optimization = Optimization(
        radius, count, width, relay_constraint, clustering=clustering)
    Problem, time_taken = optimization.solve_problem()

    # Initialize energy calculation
    energy_calculation = EnergyCalculation(
        optimization.sensorList, optimization.relayList)
    nw_e_s = energy_calculation.init_energy_s()
    nw_e_r = energy_calculation.init_energy_r()

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

    state_s = sim_instance.state_matrix()
    final_energy_s, init_energy_s = sim_instance.simu_network(
        nw_e_s, nw_e_r, state_s)

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

    def run_with_interrupt(*args):
        if interrupt_handler and interrupt_handler.should_stop:
            return None
        return run_single_simulation(*args)

    def sigint_handler(signum, frame):
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

    # Store original handler
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = []
            for count in sensor_counts:
                if interrupt_handler and interrupt_handler.should_stop:
                    print("[🛑 Stopping] Interrupt received")
                    break

                future = executor.submit(
                    run_single_simulation,
                    radius, count, rounds, width,
                    relay_constraint, clustering, result_dir
                )
                futures.append((count, future))

            # Process results as they complete
            for count, future in futures:
                if interrupt_handler and interrupt_handler.check():
                    print(
                        "[🛑 Stopping] Interrupt received, cancelling remaining tasks...")
                    for future_to_cancel in futures:
                        future_to_cancel[1].cancel()
                    break

                try:
                    result = future.result(timeout=600)  # Increased timeout to 10 minutes
                    if result:
                        relay, energy, time = result
                        relays.append(relay)
                        energies.append(energy)
                        times.append(time)
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
            interrupt_handler._signal_handler(signal.SIGINT, None)

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
