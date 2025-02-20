# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from optimization import Optimization
# from simulation import Simulation
# from energy_calculation import EnergyCalculation
# from functools import lru_cache


# class NetworkEvaluation:
#     def __init__(self, config_set, result_dir):
#         self.radius = config_set["radius"]
#         self.sensor_counts = config_set["sensor_counts"]
#         self.rounds = config_set["rounds"]
#         self.width = config_set["width"]
#         self.relay_constraint = config_set["relay_constraint"]

#         # Use provided result_dir and append evaluation subfolder
#         self.results_dir = os.path.join(result_dir, "evaluation")
#         os.makedirs(self.results_dir, exist_ok=True)
#         os.makedirs(os.path.join(self.results_dir, "figures"), exist_ok=True)
#         os.makedirs(os.path.join(self.results_dir, "metrics"), exist_ok=True)

#     def run_comparative_analysis(self):
#         """Run analysis with and without clustering"""
#         metrics = {
#             'with_clustering': self._run_simulation(True),
#             'without_clustering': self._run_simulation(False)
#         }
#         self._generate_plots(metrics)
#         return metrics

#     def _run_simulation(self, clustering):
#         results = {
#             'network_lifetime': [],
#             'energy_consumption': [],
#             'alive_sensors': [],
#             'cluster_balance': [],
#             'avg_load_factor': [],
#             'energy_per_round': [],      # New metric
#             'alive_nodes_history': [],   # New metric
#             'first_death': [],           # New metric
#             'last_death': []             # New metric
#         }

#         for count in self.sensor_counts:
#             try:
#                 opt = Optimization(self.radius, count, self.width,
#                                    self.relay_constraint, clustering)
#                 energy_calc = EnergyCalculation(opt.sensorList, opt.relayList)
#                 nw_e_s = energy_calc.init_energy_s()
#                 nw_e_r = energy_calc.init_energy_r()

#                 # Initialize round-by-round tracking
#                 round_energy = []
#                 alive_nodes = []
#                 initial_total_energy = np.sum(nw_e_s)

#                 sim = Simulation(opt.sensorList, opt.relayList,
#                                  opt.connection_s_r(), opt.connection_r_r(),
#                                  opt.membership_values, opt.cluster_heads,
#                                  self.rounds, result_dir=os.path.dirname(self.results_dir))

#                 state_s = sim.state_matrix()
#                 final_energy_s, init_energy_s = sim.simu_network(
#                     nw_e_s, nw_e_r, state_s)

#                 # Calculate energy consumption per round
#                 for round_idx in range(self.rounds):
#                     current_energy = np.sum(final_energy_s)
#                     energy_consumed = initial_total_energy - current_energy
#                     round_energy.append(energy_consumed)

#                     # Count alive nodes (nodes with non-zero energy)
#                     alive_count = np.sum(np.sum(final_energy_s, axis=1) > 0)
#                     alive_nodes.append(alive_count)

#                 # Calculate first and last death rounds
#                 first_death = next((i for i, alive in enumerate(alive_nodes)
#                                     if alive < len(opt.sensorList)), self.rounds)
#                 last_death = len(opt.sensorList) - next((i for i, alive in enumerate(reversed(alive_nodes))
#                                                          if alive > 0), 0)

#                 # Store historical data
#                 results['energy_per_round'].append(round_energy)
#                 results['alive_nodes_history'].append(alive_nodes)
#                 results['first_death'].append(first_death)
#                 results['last_death'].append(last_death)

#                 # Calculate and store other metrics as before
#                 lifetime = self._calculate_network_lifetime(final_energy_s)
#                 energy_consumed = self._calculate_energy_consumption(
#                     init_energy_s, final_energy_s)
#                 alive_ratio = self._calculate_alive_sensors(final_energy_s)
#                 load_factors = energy_calc.calculate_load_balance(
#                     final_energy_s)

#                 # Store regular metrics
#                 results['network_lifetime'].append(float(lifetime))
#                 results['energy_consumption'].append(float(energy_consumed))
#                 results['alive_sensors'].append(float(alive_ratio))
#                 results['avg_load_factor'].append(float(np.mean(load_factors)))

#                 if clustering:
#                     cluster_balance = self._calculate_cluster_balance(
#                         opt.cluster_labels, final_energy_s)
#                     results['cluster_balance'].append(
#                         float(np.mean(cluster_balance)))
#                 else:
#                     results['cluster_balance'].append(0.0)

#             except Exception as e:
#                 print(
#                     f"Warning: Error in simulation for {count} sensors: {str(e)}")
#                 # Add default values for all metrics
#                 for key in results:
#                     if key in ['energy_per_round', 'alive_nodes_history']:
#                         results[key].append([0.0] * self.rounds)
#                     else:
#                         results[key].append(0.0)

#         # Convert metrics to numpy arrays
#         for key in results:
#             results[key] = np.array(results[key])

#         return results

#     def _calculate_network_lifetime(self, final_energy):
#         """Calculate network lifetime as rounds until first node dies"""
#         try:
#             # Convert to numpy array if not already
#             if not isinstance(final_energy, np.ndarray):
#                 final_energy = np.array(final_energy)

#             # Calculate sum of energy for each node
#             node_energies = np.sum(final_energy, axis=1)

#             # Return minimum energy (network lifetime)
#             return float(np.min(node_energies))
#         except Exception as e:
#             print(f"Network lifetime calculation error: {e}")
#             return 0.0

#     def _calculate_energy_consumption(self, initial_energy, final_energy):
#         """Calculate total energy consumed"""
#         return np.sum(initial_energy) - np.sum(final_energy)

#     def _calculate_alive_sensors(self, final_energy):
#         """Calculate ratio of alive sensors"""
#         total = len(final_energy)
#         alive = sum(1 for row in final_energy if np.sum(row) > 0)
#         return alive / total

#     def _calculate_cluster_balance(self, cluster_labels, final_energy):
#         """Calculate energy balance among clusters"""
#         unique_clusters = np.unique(cluster_labels)
#         cluster_energies = []

#         for cluster in unique_clusters:
#             cluster_nodes = np.where(cluster_labels == cluster)[0]
#             cluster_energy = np.mean([np.sum(final_energy[i])
#                                      for i in cluster_nodes])
#             cluster_energies.append(cluster_energy)

#         # Return numpy array instead of module
#         return np.array(cluster_energies)

#     def _generate_plots(self, metrics):
#         """Generate individual, clear visualizations"""
#         os.makedirs(f"{self.results_dir}/figures", exist_ok=True)

#         # Network Performance Metrics
#         self._plot_network_lifetime_bar(metrics)
#         self._plot_alive_sensors_bar(metrics)
#         self._plot_energy_consumption_bar(metrics)
#         self._plot_load_balance_bar(metrics)

#         # Time Series Analysis
#         self._plot_energy_over_time(metrics)
#         self._plot_node_survival_over_time(metrics)

#         # Comparative Analysis
#         self._plot_clustering_impact(metrics)
#         self._plot_energy_efficiency(metrics)

#         # Save metrics to CSV
#         self._save_metrics(metrics)

#     def _plot_network_lifetime_bar(self, metrics):
#         """Bar plot of network lifetime comparison"""
#         plt.figure(figsize=(12, 6))
#         x = np.arange(len(self.sensor_counts))
#         width = 0.35

#         plt.bar(x - width/2, metrics['with_clustering']['network_lifetime'],
#                 width, label='With Clustering', color='skyblue')
#         plt.bar(x + width/2, metrics['without_clustering']['network_lifetime'],
#                 width, label='Without Clustering', color='lightcoral')

#         plt.xlabel('Number of Sensors')
#         plt.ylabel('Network Lifetime (rounds)')
#         plt.title('Network Lifetime Comparison')
#         plt.xticks(x, self.sensor_counts)
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.savefig(
#             f"{self.results_dir}/figures/network_lifetime_bar.png", bbox_inches='tight')
#         plt.close()

#     def _plot_energy_consumption_bar(self, metrics):
#         """Bar plot of energy consumption"""
#         plt.figure(figsize=(12, 6))
#         x = np.arange(len(self.sensor_counts))
#         width = 0.35

#         plt.bar(x - width/2, metrics['with_clustering']['energy_consumption'],
#                 width, label='With Clustering', color='forestgreen')
#         plt.bar(x + width/2, metrics['without_clustering']['energy_consumption'],
#                 width, label='Without Clustering', color='indianred')

#         plt.xlabel('Number of Sensors')
#         plt.ylabel('Energy Consumed')
#         plt.title('Energy Consumption Comparison')
#         plt.xticks(x, self.sensor_counts)
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.savefig(
#             f"{self.results_dir}/figures/energy_consumption_bar.png", bbox_inches='tight')
#         plt.close()

#     def _plot_energy_over_time(self, metrics):
#         """Line plot of energy consumption over time"""
#         plt.figure(figsize=(12, 6))
#         rounds = range(self.rounds)

#         for count_idx, count in enumerate(self.sensor_counts):
#             plt.plot(rounds, metrics['with_clustering']['energy_per_round'][count_idx],
#                      label=f'{count} Sensors (With Clustering)')

#         plt.xlabel('Round Number')
#         plt.ylabel('Energy Consumed')
#         plt.title('Energy Consumption Over Time')
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(
#             f"{self.results_dir}/figures/energy_over_time.png", bbox_inches='tight')
#         plt.close()

#     def _plot_node_survival_over_time(self, metrics):
#         """Line plot of node survival rate"""
#         plt.figure(figsize=(12, 6))
#         rounds = range(self.rounds)

#         for count in self.sensor_counts:
#             plt.plot(rounds,
#                      [n/count * 100 for n in metrics['with_clustering']
#                          ['alive_nodes_history'][0]],
#                      label=f'{count} Sensors')

#         plt.xlabel('Round Number')
#         plt.ylabel('Survival Rate (%)')
#         plt.title('Node Survival Rate Over Time')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(
#             f"{self.results_dir}/figures/node_survival_rate.png", bbox_inches='tight')
#         plt.close()

#     def _plot_clustering_impact(self, metrics):
#         """Bar plot showing clustering impact on different metrics"""
#         plt.figure(figsize=(12, 6))
#         x = np.arange(len(self.sensor_counts))
#         width = 0.35

#         improvement = ((metrics['with_clustering']['network_lifetime'] -
#                        metrics['without_clustering']['network_lifetime']) /
#                        metrics['without_clustering']['network_lifetime'] * 100)

#         plt.bar(x, improvement, width, color='lightseagreen')
#         plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

#         plt.xlabel('Number of Sensors')
#         plt.ylabel('Improvement (%)')
#         plt.title('Impact of Clustering on Network Performance')
#         plt.xticks(x, self.sensor_counts)
#         plt.grid(True, alpha=0.3)
#         plt.savefig(
#             f"{self.results_dir}/figures/clustering_impact.png", bbox_inches='tight')
#         plt.close()

#     def _calculate_metrics_history(self, simulation_results):
#         """Calculate historical metrics for each round"""
#         try:
#             rounds = range(self.rounds)
#             energy_per_round = []
#             alive_nodes_history = []

#             for round_num in rounds:
#                 energy = simulation_results[f'round_{round_num}_energy']
#                 alive = simulation_results[f'round_{round_num}_alive']
#                 energy_per_round.append(energy)
#                 alive_nodes_history.append(alive)

#             # Calculate first death - when first node dies
#             try:
#                 first_death = next(i for i, alive in enumerate(alive_nodes_history)
#                                    if alive < len(self.sensorList))
#             except StopIteration:
#                 first_death = self.rounds

#             # Calculate last death - when last node dies
#             try:
#                 reversed_enum = list(enumerate(reversed(alive_nodes_history)))
#                 last_death = next(i for i, alive in reversed_enum if alive > 0)
#             except StopIteration:
#                 last_death = 0

#             return {
#                 'energy_per_round': energy_per_round,
#                 'alive_nodes_history': alive_nodes_history,
#                 'first_death': first_death,
#                 'last_death': last_death
#             }

#         except Exception as e:
#             print(f"Error in metrics history calculation: {e}")
#             return {
#                 'energy_per_round': [0] * self.rounds,
#                 'alive_nodes_history': [0] * self.rounds,
#                 'first_death': 0,
#                 'last_death': 0
#             }

#     def _save_summary(self, metrics):
#         """Save comprehensive summary to text file"""
#         with open(os.path.join(self.results_dir, "summary.txt"), 'w') as f:
#             f.write("Network Performance Analysis\n")
#             f.write("==========================\n\n")

#             # Network Lifetime
#             f.write("Network Lifetime:\n")
#             f.write(
#                 f"With Clustering: {np.mean(metrics['with_clustering']['network_lifetime']):.2f} rounds\n")
#             f.write(
#                 f"Without Clustering: {np.mean(metrics['without_clustering']['network_lifetime']):.2f} rounds\n\n")

#             # Energy Efficiency
#             f.write("Energy Consumption:\n")
#             f.write(
#                 f"With Clustering: {np.mean(metrics['with_clustering']['energy_consumption']):.2f} units\n")
#             f.write(
#                 f"Without Clustering: {np.mean(metrics['without_clustering']['energy_consumption']):.2f} units\n\n")

#             # Load Balance
#             f.write("Load Balance:\n")
#             f.write(
#                 f"With Clustering: {np.mean(metrics['with_clustering']['avg_load_factor']):.4f}\n")
#             f.write(
#                 f"Without Clustering: {np.mean(metrics['without_clustering']['avg_load_factor']):.4f}\n\n")

#             # Cluster Analysis (only for clustering enabled)
#             if np.any(metrics['with_clustering']['cluster_balance'] > 0):
#                 f.write("Cluster Balance:\n")
#                 f.write(
#                     f"Average: {np.mean(metrics['with_clustering']['cluster_balance']):.4f}\n")
#                 f.write(
#                     f"Std Dev: {np.std(metrics['with_clustering']['cluster_balance']):.4f}\n")

#     def _plot_alive_sensors_bar(self, metrics):
#         """Bar plot of alive sensors ratio"""
#         plt.figure(figsize=(12, 6))
#         x = np.arange(len(self.sensor_counts))
#         width = 0.35

#         plt.bar(x - width/2, metrics['with_clustering']['alive_sensors'] * 100,
#                 width, label='With Clustering', color='mediumseagreen')
#         plt.bar(x + width/2, metrics['without_clustering']['alive_sensors'] * 100,
#                 width, label='Without Clustering', color='salmon')

#         plt.xlabel('Number of Sensors')
#         plt.ylabel('Alive Sensors (%)')
#         plt.title('Sensor Survival Rate Comparison')
#         plt.xticks(x, self.sensor_counts)
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.savefig(
#             f"{self.results_dir}/figures/alive_sensors_bar.png", bbox_inches='tight')
#         plt.close()

#     def _plot_load_balance_bar(self, metrics):
#         """Bar plot of load balancing metrics"""
#         plt.figure(figsize=(12, 6))
#         x = np.arange(len(self.sensor_counts))
#         width = 0.35

#         plt.bar(x - width/2, metrics['with_clustering']['avg_load_factor'],
#                 width, label='With Clustering', color='cornflowerblue')
#         plt.bar(x + width/2, metrics['without_clustering']['avg_load_factor'],
#                 width, label='Without Clustering', color='lightcoral')

#         plt.xlabel('Number of Sensors')
#         plt.ylabel('Average Load Factor')
#         plt.title('Load Balance Distribution')
#         plt.xticks(x, self.sensor_counts)
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.savefig(
#             f"{self.results_dir}/figures/load_balance_bar.png", bbox_inches='tight')
#         plt.close()

#     def _plot_energy_efficiency(self, metrics):
#         """Plot energy efficiency comparison"""
#         plt.figure(figsize=(12, 6))
#         x = np.arange(len(self.sensor_counts))
#         width = 0.35

#         # Calculate energy efficiency (network lifetime / energy consumed)
#         efficiency_with = metrics['with_clustering']['network_lifetime'] / \
#             (metrics['with_clustering']['energy_consumption'] + 1e-10)
#         efficiency_without = metrics['without_clustering']['network_lifetime'] / \
#             (metrics['without_clustering']['energy_consumption'] + 1e-10)

#         plt.bar(x - width/2, efficiency_with,
#                 width, label='With Clustering', color='lightgreen')
#         plt.bar(x + width/2, efficiency_without,
#                 width, label='Without Clustering', color='lightpink')

#         plt.xlabel('Number of Sensors')
#         plt.ylabel('Energy Efficiency (Lifetime/Energy)')
#         plt.title('Energy Efficiency Comparison')
#         plt.xticks(x, self.sensor_counts)
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.savefig(
#             f"{self.results_dir}/figures/energy_efficiency.png", bbox_inches='tight')
#         plt.close()

#     def _save_metrics(self, metrics):
#         """Save all metrics to CSV files in the metrics directory"""
#         try:
#             metrics_dir = os.path.join(self.results_dir, "metrics")
#             os.makedirs(metrics_dir, exist_ok=True)

#             # Core metrics to save
#             core_metrics = [
#                 'network_lifetime',
#                 'energy_consumption',
#                 'alive_sensors',
#                 'avg_load_factor'
#             ]

#             # Save each metric for both clustering and non-clustering cases
#             for method in ['with_clustering', 'without_clustering']:
#                 for metric_name in core_metrics:
#                     if metric_name in metrics[method]:
#                         filename = os.path.join(
#                             metrics_dir, f"{method}_{metric_name}.csv")
#                         try:
#                             # Create data array with sensor counts
#                             data = np.column_stack((
#                                 self.sensor_counts,
#                                 metrics[method][metric_name]
#                             ))
#                             # Save with header
#                             np.savetxt(
#                                 filename,
#                                 data,
#                                 delimiter=',',
#                                 header='sensor_count,' + metric_name,
#                                 comments='',
#                                 fmt=['%d', '%.6f']
#                             )
#                         except Exception as e:
#                             print(
#                                 f"Warning: Could not save {metric_name} for {method}: {str(e)}")

#             # Save clustering-specific metrics
#             if 'with_clustering' in metrics:
#                 if 'cluster_balance' in metrics['with_clustering']:
#                     filename = os.path.join(
#                         metrics_dir, "clustering_balance.csv")
#                     try:
#                         data = np.column_stack((
#                             self.sensor_counts,
#                             metrics['with_clustering']['cluster_balance']
#                         ))
#                         np.savetxt(
#                             filename,
#                             data,
#                             delimiter=',',
#                             header='sensor_count,cluster_balance',
#                             comments='',
#                             fmt=['%d', '%.6f']
#                         )
#                     except Exception as e:
#                         print(
#                             f"Warning: Could not save cluster balance metrics: {str(e)}")

#         except Exception as e:
#             print(f"Error in saving metrics: {str(e)}")
