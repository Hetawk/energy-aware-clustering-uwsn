import matplotlib.pyplot as plt
import numpy as np


class EnergyPlots:
    @staticmethod
    def plot_energy_consumption(sensor_counts, metrics, save_path):
        plt.figure(figsize=(12, 6))
        x = np.arange(len(sensor_counts))
        width = 0.35

        plt.bar(x - width/2, metrics['with_clustering']['energy_consumption'],
                width, label='With Clustering', color='forestgreen')
        plt.bar(x + width/2, metrics['without_clustering']['energy_consumption'],
                width, label='Without Clustering', color='indianred')

        plt.xlabel('Number of Sensors')
        plt.ylabel('Energy Consumed')
        plt.title('Energy Consumption Comparison')
        plt.xticks(x, sensor_counts)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_energy_efficiency(sensor_counts, metrics, save_path):
        plt.figure(figsize=(12, 6))
        x = np.arange(len(sensor_counts))
        width = 0.35

        efficiency_with = metrics['with_clustering']['network_lifetime'] / \
            (metrics['with_clustering']['energy_consumption'] + 1e-10)
        efficiency_without = metrics['without_clustering']['network_lifetime'] / \
            (metrics['without_clustering']['energy_consumption'] + 1e-10)

        plt.bar(x - width/2, efficiency_with,
                width, label='With Clustering', color='lightgreen')
        plt.bar(x + width/2, efficiency_without,
                width, label='Without Clustering', color='lightpink')

        plt.xlabel('Number of Sensors')
        plt.ylabel('Energy Efficiency (Lifetime/Energy)')
        plt.title('Energy Efficiency Comparison')
        plt.xticks(x, sensor_counts)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
