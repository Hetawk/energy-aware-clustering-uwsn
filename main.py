# main.py
import os
import csv
import json
import argparse
import matplotlib.pyplot as plt
from optimization import Optimization
from simulation import Simulation
from energy_calculation import EnergyCalculation

# Ensure the results directory exists
os.makedirs("results", exist_ok=True)

def save_to_csv(filename, headers, data):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

def run_simulations(radius, sensor_counts, rounds, width, relay_constraint, clustering):
    relays = []
    energies = []
    times = []  # To store the time taken for each simulation
    for count in sensor_counts:
        optimization = Optimization(radius, count, width, relay_constraint, clustering=clustering)
        Problem, time_taken = optimization.solve_problem()
        times.append(time_taken)
        energy_calculation = EnergyCalculation(optimization.sensorList, optimization.relayList)
        nw_e_s = energy_calculation.init_energy_s()
        nw_e_r = energy_calculation.init_energy_r()
        state_s = Simulation(optimization.sensorList, optimization.relayList, optimization.connection_s_r(),
                             optimization.connection_r_r(), optimization.membership_values,
                             optimization.cluster_heads).state_matrix()
        simulation = Simulation(optimization.sensorList, optimization.relayList, optimization.connection_s_r(),
                                optimization.connection_r_r(), optimization.membership_values,
                                optimization.cluster_heads, rounds)
        final_energy_s, init_energy_s = simulation.simu_network(nw_e_s, nw_e_r, state_s)
        # Calculate consumed energy based on the preserved initial energy matrix
        consumed = sum(sum(row) for row in init_energy_s) - sum(sum(row) for row in final_energy_s)
        relays.append(len(optimization.relayList))
        energies.append(consumed)

    # Save results to CSV
    save_to_csv("results/network/relays.csv", ["Sensor Count", "Relays"], zip(sensor_counts, relays))
    save_to_csv("results/energy/energies.csv", ["Sensor Count", "Energy"], zip(sensor_counts, energies))
    save_to_csv("results/evaluation/metrics/times.csv", ["Sensor Count", "Time Taken"], zip(sensor_counts, times))

    return relays, energies, times

def main():
    parser = argparse.ArgumentParser(description="Run network simulations.")
    parser.add_argument("--config", type=str, help="Path to the configuration file", default="config.json")
    parser.add_argument("--set", type=str, help="Configuration set to use", default="default")
    parser.add_argument("--clustering", action='store_true', help="Enable clustering")
    parser.add_argument("--no-clustering", action='store_false', help="Disable clustering")
    parser.set_defaults(clustering=True)
    args = parser.parse_args()

    # Create all required directories at startup
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/network", exist_ok=True)
    os.makedirs("results/energy", exist_ok=True)
    os.makedirs("results/state", exist_ok=True)
    os.makedirs("results/evaluation", exist_ok=True)
    os.makedirs("results/evaluation/figures", exist_ok=True)
    os.makedirs("results/evaluation/metrics", exist_ok=True)

    with open(args.config, "r") as file:
        config = json.load(file)

    config_set = config["configurations"][args.set]

    radius = config_set["radius"]
    sensor_counts = config_set["sensor_counts"]
    rounds = config_set["rounds"]
    width = config_set["width"]
    relay_constraint = config_set["relay_constraint"]

    # Run simulations with the specified clustering setting
    relays, energies, times = run_simulations(radius, sensor_counts, rounds, width, relay_constraint, clustering=args.clustering)

    # Log the time taken for each simulation
    print(f"Time taken: {times}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(sensor_counts, energies, label='Energy Consumption', marker='o')
    plt.title('Energy Consumption')
    plt.xlabel('Number of Sensors')
    plt.ylabel('Energy Used')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/evaluation/figures/energy_comparison.png")
    # plt.show()  # Remove this line to prevent the graph from popping up

if __name__ == "__main__":
    main()