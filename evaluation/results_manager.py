import os
import numpy as np
from visual.network_plots import NetworkPlots
from visual.energy_plots import EnergyPlots
from visual.load_plots import LoadPlots


class ResultsManager:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, "figures")
        self.metrics_dir = os.path.join(results_dir, "metrics")
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def generate_all_plots(self, metrics, sensor_counts, rounds):
        """Generate all visualization plots"""
        # Network plots
        NetworkPlots.plot_network_lifetime(
            sensor_counts, metrics,
            os.path.join(self.figures_dir, "network_lifetime_bar.png")
        )
        NetworkPlots.plot_node_survival(
            sensor_counts, metrics, rounds,
            os.path.join(self.figures_dir, "node_survival_rate.png")
        )

        # Energy plots
        EnergyPlots.plot_energy_consumption(
            sensor_counts, metrics,
            os.path.join(self.figures_dir, "energy_consumption_bar.png")
        )
        EnergyPlots.plot_energy_efficiency(
            sensor_counts, metrics,
            os.path.join(self.figures_dir, "energy_efficiency.png")
        )

        # Load plots
        LoadPlots.plot_load_balance(
            sensor_counts, metrics,
            os.path.join(self.figures_dir, "load_balance_bar.png")
        )

    def save_metrics(self, metrics, sensor_counts):
        """Save metrics to CSV files"""
        try:
            os.makedirs(self.metrics_dir, exist_ok=True)

            # Core metrics to save
            core_metrics = [
                'network_lifetime',
                'energy_consumption',
                'alive_sensors',
                'avg_load_factor'
            ]

            # Save each metric for both clustering and non-clustering cases
            for method in ['with_clustering', 'without_clustering']:
                for metric_name in core_metrics:
                    if metric_name in metrics[method]:
                        filename = os.path.join(
                            self.metrics_dir, f"{method}_{metric_name}.csv")
                        try:
                            data = np.column_stack((
                                sensor_counts,
                                metrics[method][metric_name]
                            ))
                            np.savetxt(
                                filename,
                                data,
                                delimiter=',',
                                header='sensor_count,' + metric_name,
                                comments='',
                                fmt=['%d', '%.6f']
                            )
                        except Exception as e:
                            print(
                                f"Warning: Could not save {metric_name} for {method}: {str(e)}")

            # Save per-round metrics
            for method in ['with_clustering', 'without_clustering']:
                if 'energy_per_round' in metrics[method]:
                    filename = os.path.join(
                        self.metrics_dir, f"{method}_energy_per_round.csv")
                    try:
                        np.savetxt(filename, metrics[method]['energy_per_round'],
                                   delimiter=',', fmt='%.6f')
                    except Exception as e:
                        print(
                            f"Warning: Could not save energy_per_round for {method}: {str(e)}")

        except Exception as e:
            print(f"Error in saving metrics: {str(e)}")

    def save_summary(self, metrics):
        """Save comprehensive summary to text file"""
        try:
            summary_path = os.path.join(self.results_dir, "summary.txt")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)

            with open(summary_path, 'w') as f:
                f.write("Network Performance Analysis\n")
                f.write("==========================\n\n")

                for method in ['with_clustering', 'without_clustering']:
                    f.write(f"\n{method.replace('_', ' ').title()}:\n")
                    f.write("-" * (len(method) + 1) + "\n")

                    # Network Lifetime
                    lifetime = np.mean(metrics[method]['network_lifetime'])
                    f.write(
                        f"Average Network Lifetime: {lifetime:.2f} rounds\n")

                    # Energy Consumption
                    energy = np.mean(metrics[method]['energy_consumption'])
                    f.write(
                        f"Average Energy Consumption: {energy:.2f} units\n")

                    # Load Balance
                    if 'avg_load_factor' in metrics[method]:
                        load = np.mean(metrics[method]['avg_load_factor'])
                        f.write(f"Average Load Factor: {load:.4f}\n")

        except Exception as e:
            print(f"Error saving summary: {str(e)}")

    def save_partial_results(self, metrics, sensor_counts):
        """Save partial simulation results when interrupted"""
        partial_file = os.path.join(self.results_dir, "partial_results.txt")
        try:
            with open(partial_file, "w") as f:
                f.write("Partial Results:\n")
                f.write(f"Sensor Counts: {sensor_counts}\n")
                f.write(f"Metrics: {str(metrics)}\n")
            print("Partial results saved to:", partial_file)
        except Exception as e:
            print("Failed to save partial results:", e)
