# main.py
import os
import csv
import json
import argparse
import matplotlib.pyplot as plt
from optimization import Optimization
from simulation import Simulation
from energy_calculation import EnergyCalculation
from validate_implementation import validate_implementation


def save_to_csv(filename, headers, data):
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)


def run_simulations(radius, sensor_counts, rounds, width, relay_constraint, clustering, result_dir):
    """Run simulations with all output going to config-specific directory"""
    relays = []
    energies = []
    times = []

    # Create all directories under the config-specific folder
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "network"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "energy"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "state"), exist_ok=True)
    os.makedirs(os.path.join(
        result_dir, "evaluation", "metrics"), exist_ok=True)
    os.makedirs(os.path.join(
        result_dir, "evaluation", "figures"), exist_ok=True)

    for count in sensor_counts:
        optimization = Optimization(
            radius, count, width, relay_constraint, clustering=clustering)
        Problem, time_taken = optimization.solve_problem()
        times.append(time_taken)

        # Initialize energy calculation
        energy_calculation = EnergyCalculation(
            optimization.sensorList, optimization.relayList)
        nw_e_s = energy_calculation.init_energy_s()
        nw_e_r = energy_calculation.init_energy_r()

        # Always pass the config-specific result_dir
        sim_instance = Simulation(optimization.sensorList, optimization.relayList,
                                  optimization.connection_s_r(), optimization.connection_r_r(),
                                  optimization.membership_values, optimization.cluster_heads,
                                  rounds, result_dir=result_dir)

        state_s = sim_instance.state_matrix()
        final_energy_s, init_energy_s = sim_instance.simu_network(
            nw_e_s, nw_e_r, state_s)

        consumed = sum(sum(row) for row in init_energy_s) - \
            sum(sum(row) for row in final_energy_s)
        relays.append(len(optimization.relayList))
        energies.append(consumed)

    # Save all results under config directory
    save_to_csv(os.path.join(result_dir, "network", "relays.csv"),
                ["Sensor Count", "Relays"], zip(sensor_counts, relays))
    save_to_csv(os.path.join(result_dir, "energy", "energies.csv"),
                ["Sensor Count", "Energy"], zip(sensor_counts, energies))
    save_to_csv(os.path.join(result_dir, "evaluation", "metrics", "times.csv"),
                ["Sensor Count", "Time Taken"], zip(sensor_counts, times))

    return relays, energies, times


def main():
    parser = argparse.ArgumentParser(
        description="Run network simulations and validation.")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to configuration file")
    parser.add_argument("--set", type=str, default="default",
                        help="Configuration set to use")
    parser.add_argument("--clustering", action='store_true',
                        help="Enable clustering")
    parser.add_argument("--no-clustering", action='store_false',
                        dest='clustering', help="Disable clustering")
    parser.add_argument("--mode", type=str, choices=['simulate', 'validate', 'both'],
                        default='both', help="Run mode: simulate, validate, or both")
    parser.set_defaults(clustering=True)
    args = parser.parse_args()

    # Create only the config-specific results directory
    result_dir = os.path.join("results", args.set)

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    config_set = config["configurations"][args.set]

    # Run simulations if requested
    if args.mode in ['simulate', 'both']:
        # For 'both' mode, we need to run with and without clustering
        if args.mode == 'both':
            # First run with clustering enabled
            print("\nRunning simulation with clustering...")
            relays_c, energies_c, times_c = run_simulations(
                config_set["radius"],
                config_set["sensor_counts"],
                config_set["rounds"],
                config_set["width"],
                config_set["relay_constraint"],
                clustering=True,
                result_dir=result_dir
            )
            print(f"Time taken (clustering): {times_c}")

            # Then run without clustering
            print("\nRunning simulation without clustering...")
            relays_nc, energies_nc, times_nc = run_simulations(
                config_set["radius"],
                config_set["sensor_counts"],
                config_set["rounds"],
                config_set["width"],
                config_set["relay_constraint"],
                clustering=False,
                result_dir=result_dir
            )
            print(f"Time taken (no clustering): {times_nc}")

        else:
            # Single mode - use the clustering flag as specified
            relays, energies, times = run_simulations(
                config_set["radius"],
                config_set["sensor_counts"],
                config_set["rounds"],
                config_set["width"],
                config_set["relay_constraint"],
                clustering=args.clustering,
                result_dir=result_dir
            )
            print(f"Time taken: {times}")

        # Generate comparison plots if we have both results
        if args.mode == 'both':
            plt.figure(figsize=(10, 6))
            plt.plot(config_set["sensor_counts"], energies_c,
                     label='With Clustering', marker='o')
            plt.plot(config_set["sensor_counts"], energies_nc,
                     label='Without Clustering', marker='s')
            plt.title('Energy Consumption Comparison')
            plt.xlabel('Number of Sensors')
            plt.ylabel('Energy Used')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, "evaluation",
                        "figures", "energy_comparison.png"))
            plt.close()
        else:
            # Single mode plot
            plt.figure(figsize=(10, 6))
            plt.plot(config_set["sensor_counts"], energies,
                     label='Energy Consumption', marker='o')
            plt.title('Energy Consumption')
            plt.xlabel('Number of Sensors')
            plt.ylabel('Energy Used')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(result_dir, "evaluation",
                        "figures", "energy_comparison.png"))
            plt.close()

    # Run validation if requested
    if args.mode in ['validate', 'both']:
        validation_results = validate_implementation(args.config, args.set)


if __name__ == "__main__":
    main()
