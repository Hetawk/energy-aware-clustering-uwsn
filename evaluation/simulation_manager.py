from optimization import Optimization
from energy_calculation import EnergyCalculation
from simulation import Simulation
import numpy as np
import os
from utils.interrupt_handler import GracefulInterruptHandler


class SimulationManager:
    def __init__(self, evaluation):
        self.evaluation = evaluation

    def run_simulation(self, clustering):
        """Run simulation with specified clustering setting"""
        results = self._initialize_results()

        with GracefulInterruptHandler() as handler:
            for count in self.evaluation.sensor_counts:
                if handler.should_stop:
                    print(f"\n[📊 Results] Saving partial results...")
                    break

                try:
                    print(f"\n[🔄 Running] Processing {count} sensors")
                    final_metrics = self._run_single_simulation(
                        count, clustering, self.evaluation.radius,
                        self.evaluation.width, self.evaluation.relay_constraint,
                        self.evaluation.rounds
                    )
                    self._store_simulation_results(results, final_metrics)
                    print(
                        f"[✅ Complete] Simulation completed for {count} sensors")

                except Exception as e:
                    print(
                        f"[❌ Error] Simulation failed for {count} sensors: {str(e)}")
                    self._store_default_results(results)

        return results

    @staticmethod
    def _initialize_results():
        return {
            'network_lifetime': [],
            'energy_consumption': [],
            'alive_sensors': [],
            'cluster_balance': [],
            'avg_load_factor': [],
            'energy_per_round': [],
            'alive_nodes_history': [],
            'first_death': [],
            'last_death': []
        }

    def _run_single_simulation(self, count, clustering, radius, width, relay_constraint, rounds):
        """Run a single simulation and return metrics"""
        # Create required directories first
        os.makedirs(os.path.join(
            self.evaluation.results_dir, "network"), exist_ok=True)
        os.makedirs(os.path.join(
            self.evaluation.results_dir, "energy"), exist_ok=True)
        os.makedirs(os.path.join(
            self.evaluation.results_dir, "state"), exist_ok=True)

        # Initialize energy calculations
        opt = Optimization(radius, count, width,
                           relay_constraint, clustering=clustering)
        energy_calc = EnergyCalculation(opt.sensorList, opt.relayList)
        nw_e_s = energy_calc.init_energy_s()
        # Removed nw_e_r initialization

        sim = Simulation(opt.sensorList, opt.relayList,
                         opt.connection_s_r(), opt.connection_r_r(),
                         opt.membership_values, opt.cluster_heads,
                         rounds, result_dir=self.evaluation.results_dir)

        # Update: Unpack four values from simu_network
        final_energy_s, init_energy_s, first_death, last_death = sim.simu_network(
            nw_e_s)
        return {
            'final_energy': final_energy_s,
            'initial_energy': init_energy_s,
            'first_death': first_death,
            'last_death': last_death,
            'opt': opt,
            'energy_calc': energy_calc
        }

    def _store_simulation_results(self, results, metrics):
        """Store simulation results in the results dictionary"""
        try:
            # Basic metrics
            results['network_lifetime'].append(
                self.evaluation.metrics_calculator.calculate_network_lifetime(metrics['final_energy']))
            results['energy_consumption'].append(
                self.evaluation.metrics_calculator.calculate_energy_consumption(
                    metrics['initial_energy'], metrics['final_energy']))
            results['alive_sensors'].append(
                self.evaluation.metrics_calculator.calculate_alive_sensors(metrics['final_energy']))

            # Calculate load factors
            load_factors = metrics['energy_calc'].calculate_load_balance(
                metrics['final_energy'])
            results['avg_load_factor'].append(float(np.mean(load_factors)))

            # Historical data
            alive_history = []
            energy_history = []
            for round_idx in range(self.evaluation.rounds):
                alive_count = np.sum(
                    np.sum(metrics['final_energy'], axis=1) > 0)
                energy_used = np.sum(
                    metrics['initial_energy']) - np.sum(metrics['final_energy'])
                alive_history.append(alive_count)
                energy_history.append(energy_used)

            results['alive_nodes_history'].append(alive_history)
            results['energy_per_round'].append(energy_history)

            # Cluster metrics
            if hasattr(metrics['opt'], 'cluster_labels'):
                cluster_balance = self.evaluation.metrics_calculator.calculate_cluster_balance(
                    metrics['opt'].cluster_labels, metrics['final_energy'])
                results['cluster_balance'].append(
                    float(np.mean(cluster_balance)))
            else:
                results['cluster_balance'].append(0.0)

        except Exception as e:
            print(f"Error storing simulation results: {e}")
            self._store_default_results(results)

    def _store_default_results(self, results):
        """Store default values when simulation fails"""
        for key in results:
            if key in ['energy_per_round', 'alive_nodes_history']:
                results[key].append([0.0] * self.evaluation.rounds)
            else:
                results[key].append(0.0)
