import os
import numpy as np
from evaluation.results_manager import ResultsManager
from .metrics_calculator import MetricsCalculator
from .simulation_manager import SimulationManager
import logging

# # Optionally, if MetricsSaver is used elsewhere:
# from visual.metrics_saver import MetricsSaver
# # The following imports are used by ResultsManager (they can be duplicated in that file)
# from visual.network_plots import NetworkPlots
# from visual.energy_plots import EnergyPlots
# from visual.load_plots import LoadPlots


class NetworkEvaluation:
    def __init__(self, config_dict, result_dir):
        self.radius = config_dict["radius"]
        self.sensor_counts = config_dict["sensor_counts"]
        self.rounds = config_dict["rounds"]
        self.width = config_dict["width"]
        self.relay_constraint = config_dict["relay_constraint"]

        # Initialize evaluation directory
        self.results_dir = os.path.join(result_dir, "evaluation")
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize components
        self.metrics_calculator = MetricsCalculator()
        self.simulation_manager = SimulationManager(self)
        self.results_manager = ResultsManager(self.results_dir)

    def run_comparative_analysis(self):
        """Run analysis with and without clustering"""
        metrics = {}  # Ensure metrics is defined
        try:
            logging.info("Beginning comparative analysis...")
            metrics = {
                'with_clustering': self.simulation_manager.run_simulation(True),
                'without_clustering': self.simulation_manager.run_simulation(False)
            }
            logging.info("Generating plots and saving metrics...")
            # Generate all plots using the results manager
            self.results_manager.generate_all_plots(
                metrics, self.sensor_counts, self.rounds)
            self.results_manager.save_metrics(metrics, self.sensor_counts)
            self.results_manager.save_summary(metrics)
            logging.info("Analysis completed successfully")
            return metrics
        except KeyboardInterrupt:
            logging.error("Analysis interrupted by user")
            if metrics:
                self.results_manager.save_partial_results(
                    metrics, self.sensor_counts)
            return None
