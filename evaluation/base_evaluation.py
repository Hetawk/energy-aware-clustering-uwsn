import os
from .metrics_calculator import MetricsCalculator
from .simulation_manager import SimulationManager
from .results_manager import ResultsManager


class NetworkEvaluation:
    def __init__(self, config_set, result_dir):
        self.radius = config_set["radius"]
        self.sensor_counts = config_set["sensor_counts"]
        self.rounds = config_set["rounds"]
        self.width = config_set["width"]
        self.relay_constraint = config_set["relay_constraint"]

        # Initialize result directory
        self.results_dir = os.path.join(result_dir, "evaluation")
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize components
        self.metrics_calculator = MetricsCalculator()
        self.simulation_manager = SimulationManager(self)
        self.results_manager = ResultsManager(self.results_dir)

    def run_comparative_analysis(self):
        """Run analysis with and without clustering"""
        try:
            print("\n[🚀 Starting] Beginning comparative analysis...")
            metrics = {
                'with_clustering': self.simulation_manager.run_simulation(True),
                'without_clustering': self.simulation_manager.run_simulation(False)
            }

            print("\n[📊 Processing] Generating visualizations and saving results...")
            # Generate visualizations and save results
            self.results_manager.generate_all_plots(
                metrics, self.sensor_counts, self.rounds)
            self.results_manager.save_metrics(metrics, self.sensor_counts)
            self.results_manager.save_summary(metrics)

            print("[✅ Complete] Analysis completed successfully")
            return metrics

        except KeyboardInterrupt:
            print("\n[🛑 Interrupted] Analysis interrupted by user")
            if 'metrics' in locals():
                print("[💾 Saving] Saving partial results...")
                self.results_manager.save_partial_results(
                    metrics, self.sensor_counts)
            return None
