# main.py
import os
import csv
import json
import argparse
import sys
import matplotlib.pyplot as plt
from optimization import Optimization
from simulation import Simulation
from energy_calculation import EnergyCalculation
from utils.gpu_accelerator import GPUAccelerator
from utils.interrupt_handler import GracefulInterruptHandler
from validate_implementation import validate_implementation
from evaluation import NetworkEvaluation  # Updated import
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import signal


def save_to_csv(filename, headers, data):
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)


def run_single_simulation(radius, count, rounds, width, relay_constraint, clustering, result_dir):
    optimization = Optimization(
        radius, count, width, relay_constraint, clustering=clustering)
    Problem, time_taken = optimization.solve_problem()

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
    relays = len(optimization.relayList)
    energy = consumed

    return relays, energy, time_taken


def run_simulations(radius, sensor_counts, rounds, width, relay_constraint, clustering, result_dir, interrupt_handler):
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

    # Parallel processing for multiple sensor counts
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [
            executor.submit(
                run_single_simulation,
                radius, count, rounds, width,
                relay_constraint, clustering, result_dir
            ) for count in sensor_counts
        ]
        results = [f.result() for f in futures]

    relays, energies, times = zip(*results)

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

    try:
        def handle_interrupt(signum, frame):
            print("\n\n[⚠️ Interrupt] Caught Ctrl+C...")
            answer = input(
                "Do you want to stop the program? (y/N): ").strip().lower()
            if answer == 'y':
                print("[🛑 Stopping] Gracefully shutting down...")
                sys.exit(0)
            print("[▶️ Continuing] Resuming execution...")

        # Set up interrupt handling
        signal.signal(signal.SIGINT, handle_interrupt)

        with GracefulInterruptHandler() as handler:
            # Load configuration first
            print("\n[⚙️ System] Loading configuration...")
            with open(args.config, 'r') as f:
                config = json.load(f)

            if args.set not in config["configurations"]:
                print(f"[❌ Error] Configuration set '{args.set}' not found!")
                return

            config_set = config["configurations"][args.set]

            # Print configuration details
            print(f"[⚙️ System] Configuration loaded: {args.set}")
            print(f"[⚙️ System] Sensor counts: {config_set['sensor_counts']}")
            print(f"[⚙️ System] Rounds: {config_set['rounds']}")

            # Check GPU availability
            GPUAccelerator.is_available()

            # Create result directory
            result_dir = os.path.join("results", args.set)
            print(f"[📁 System] Results will be saved to: {result_dir}")

            # Rest of the simulation code...
            if args.mode in ['simulate', 'both']:
                # For 'both' mode, we need to run with and without clustering
                if args.mode == 'both':
                    if not handler.should_stop:
                        # First run with clustering enabled
                        print("\nRunning simulation with clustering...")
                        relays_c, energies_c, times_c = run_simulations(
                            config_set["radius"],
                            config_set["sensor_counts"],
                            config_set["rounds"],
                            config_set["width"],
                            config_set["relay_constraint"],
                            clustering=True,
                            result_dir=result_dir,
                            interrupt_handler=handler
                        )
                        print(f"Time taken (clustering): {times_c}")

                    if not handler.should_stop:
                        # Then run without clustering
                        print("\nRunning simulation without clustering...")
                        relays_nc, energies_nc, times_nc = run_simulations(
                            config_set["radius"],
                            config_set["sensor_counts"],
                            config_set["rounds"],
                            config_set["width"],
                            config_set["relay_constraint"],
                            clustering=False,
                            result_dir=result_dir,
                            interrupt_handler=handler
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
                        result_dir=result_dir,
                        interrupt_handler=handler
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

            # Run validation if requested and not interrupted
            if args.mode in ['validate', 'both'] and not handler.should_stop:
                validation_results = validate_implementation(
                    args.config, args.set)

    except KeyboardInterrupt:
        print("\n[🛑 Interrupted] Program terminated by user")
        sys.exit(1)
    except FileNotFoundError:
        print(f"[❌ Error] Configuration file '{args.config}' not found!")
    except json.JSONDecodeError:
        print(f"[❌ Error] Invalid JSON in configuration file!")
    except Exception as e:
        print(f"[❌ Error] An unexpected error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
