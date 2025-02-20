import matplotlib.pyplot as plt
import numpy as np


class LoadPlots:
    @staticmethod
    def plot_load_balance(sensor_counts, metrics, save_path):
        """Plot load balance metrics with error handling"""
        plt.figure(figsize=(12, 6))
        x = np.arange(len(sensor_counts))
        width = 0.35

        # Ensure we have data to plot
        with_clustering_data = metrics['with_clustering'].get(
            'avg_load_factor', np.zeros_like(sensor_counts))
        without_clustering_data = metrics['without_clustering'].get(
            'avg_load_factor', np.zeros_like(sensor_counts))

        # Convert to numpy arrays and ensure proper shape
        with_clustering_data = np.array(with_clustering_data)
        without_clustering_data = np.array(without_clustering_data)

        if len(with_clustering_data) != len(sensor_counts):
            with_clustering_data = np.zeros_like(sensor_counts)
        if len(without_clustering_data) != len(sensor_counts):
            without_clustering_data = np.zeros_like(sensor_counts)

        plt.bar(x - width/2, with_clustering_data,
                width, label='With Clustering', color='cornflowerblue')
        plt.bar(x + width/2, without_clustering_data,
                width, label='Without Clustering', color='lightcoral')

        plt.xlabel('Number of Sensors')
        plt.ylabel('Average Load Factor')
        plt.title('Load Balance Distribution')
        plt.xticks(x, sensor_counts)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
