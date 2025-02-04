import os
import numpy as np
import matplotlib.pyplot as plt
from optimization import Optimization
from simulation import Simulation
from energy_calculation import EnergyCalculation

class NetworkEvaluation:
    def __init__(self, config_set):
        self.radius = config_set["radius"]
        self.sensor_counts = config_set["sensor_counts"]
        self.rounds = config_set["rounds"]
        self.width = config_set["width"]
        self.relay_constraint = config_set["relay_constraint"]
        self.results_dir = "results/evaluation"  # Changed from "evaluation_results" to "results/evaluation"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/figures", exist_ok=True)
        os.makedirs(f"{self.results_dir}/metrics", exist_ok=True)
        
    def run_comparative_analysis(self):
        """Run analysis with and without clustering"""
        metrics = {
            'with_clustering': self._run_simulation(True),
            'without_clustering': self._run_simulation(False)
        }
        self._generate_plots(metrics)
        return metrics

    def _run_simulation(self, clustering):
        results = {
            'network_lifetime': [],
            'energy_consumption': [],
            'alive_sensors': [],
            'cluster_balance': [],
            'avg_load_factor': []
        }
        
        for count in self.sensor_counts:
            try:
                opt = Optimization(self.radius, count, self.width, self.relay_constraint, clustering)
                energy_calc = EnergyCalculation(opt.sensorList, opt.relayList)
                nw_e_s = energy_calc.init_energy_s()
                nw_e_r = energy_calc.init_energy_r()
                
                sim = Simulation(opt.sensorList, opt.relayList, 
                               opt.connection_s_r(), opt.connection_r_r(),
                               opt.membership_values, opt.cluster_heads, self.rounds)
                
                state_s = sim.state_matrix()
                final_energy_s, init_energy_s = sim.simu_network(nw_e_s, nw_e_r, state_s)
                
                # Calculate metrics using both initial and final energy
                lifetime = self._calculate_network_lifetime(final_energy_s)
                energy_consumed = self._calculate_energy_consumption(init_energy_s, final_energy_s)
                alive_ratio = self._calculate_alive_sensors(final_energy_s)
                load_factors = energy_calc.calculate_load_balance(final_energy_s)
                
                # Store results
                results['network_lifetime'].append(float(lifetime))
                results['energy_consumption'].append(float(energy_consumed))
                results['alive_sensors'].append(float(alive_ratio))
                results['avg_load_factor'].append(float(np.mean(load_factors)))
                
                if clustering:
                    cluster_balance = self._calculate_cluster_balance(opt.cluster_labels, final_energy_s)
                    results['cluster_balance'].append(float(np.mean(cluster_balance)))
                else:
                    results['cluster_balance'].append(0.0)
                
            except Exception as e:
                print(f"Warning: Error in simulation for {count} sensors: {str(e)}")
                # Add default values to maintain array dimensions
                for key in results:
                    results[key].append(0.0)
        
        # Ensure all metrics are numpy arrays with float dtype
        for key in results:
            results[key] = np.array(results[key], dtype=float)
        
        return results

    def _calculate_network_lifetime(self, final_energy):
        """Calculate network lifetime as rounds until first node dies"""
        return np.min([np.sum(row) for row in final_energy])

    def _calculate_energy_consumption(self, initial_energy, final_energy):
        """Calculate total energy consumed"""
        return np.sum(initial_energy) - np.sum(final_energy)

    def _calculate_alive_sensors(self, final_energy):
        """Calculate ratio of alive sensors"""
        total = len(final_energy)
        alive = sum(1 for row in final_energy if np.sum(row) > 0)
        return alive / total

    def _calculate_cluster_balance(self, cluster_labels, final_energy):
        """Calculate energy balance among clusters"""
        unique_clusters = np.unique(cluster_labels)
        cluster_energies = []
        
        for cluster in unique_clusters:
            cluster_nodes = np.where(cluster_labels == cluster)[0]
            cluster_energy = np.mean([np.sum(final_energy[i]) for i in cluster_nodes])
            cluster_energies.append(cluster_energy)
            
        return np.array(cluster_energies)  # Return numpy array instead of module

    def _generate_plots(self, metrics):
        """Generate comprehensive visualization plots"""
        # Create figures directory
        os.makedirs(f"{self.results_dir}/figures", exist_ok=True)
        
        # 1. Network Lifetime Comparison
        plt.figure(figsize=(10, 6))
        plt.plot(self.sensor_counts, metrics['with_clustering']['network_lifetime'], 
                'bo-', label='With Clustering')
        plt.plot(self.sensor_counts, metrics['without_clustering']['network_lifetime'], 
                'ro-', label='Without Clustering')
        plt.title('Network Lifetime Comparison')
        plt.xlabel('Number of Sensors')
        plt.ylabel('Network Lifetime (rounds)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.results_dir}/figures/network_lifetime.png")
        plt.close()

        # 2. Energy Consumption
        plt.figure(figsize=(10, 6))
        plt.plot(self.sensor_counts, metrics['with_clustering']['energy_consumption'], 
                'g^-', label='With Clustering')
        plt.plot(self.sensor_counts, metrics['without_clustering']['energy_consumption'], 
                'm^-', label='Without Clustering')
        plt.title('Energy Consumption Comparison')
        plt.xlabel('Number of Sensors')
        plt.ylabel('Total Energy Consumed')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.results_dir}/figures/energy_consumption.png")
        plt.close()

        # 3. Load Balance Analysis
        plt.figure(figsize=(10, 6))
        plt.plot(self.sensor_counts, metrics['with_clustering']['avg_load_factor'], 
                'ks-', label='With Clustering')
        plt.plot(self.sensor_counts, metrics['without_clustering']['avg_load_factor'], 
                'ys-', label='Without Clustering')
        plt.title('Load Balance Analysis')
        plt.xlabel('Number of Sensors')
        plt.ylabel('Average Load Factor')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.results_dir}/figures/load_balance.png")
        plt.close()

        # Save metrics to CSV
        self._save_metrics(metrics)

    def _save_metrics(self, metrics):
        """Save all metrics to CSV files with proper data handling"""
        for method in ['with_clustering', 'without_clustering']:
            for metric_name, values in metrics[method].items():
                filename = f"{self.results_dir}/metrics/{method}_{metric_name}.csv"
                # Convert values to proper format before saving
                try:
                    # Convert to numpy array if not already
                    data = np.array(values, dtype=float)
                    # Add sensor counts as first column
                    save_data = np.column_stack((self.sensor_counts, data))
                    header = f"sensor_count,{metric_name}"
                    fmt = '%d,%.6f'  # Format specifier for integer,float pairs
                    np.savetxt(filename, save_data, delimiter=',', 
                             header=header, comments='', fmt=fmt)
                except Exception as e:
                    print(f"Warning: Could not save {metric_name} due to {str(e)}")
                    continue
