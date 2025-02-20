import matplotlib.pyplot as plt
import numpy as np


class NetworkPlots:
    @staticmethod
    def plot_network_lifetime(sensor_counts, metrics, save_path):
        plt.figure(figsize=(12, 6))
        x = np.arange(len(sensor_counts))
        width = 0.35

        plt.bar(x - width/2, metrics['with_clustering']['network_lifetime'],
                width, label='With Clustering', color='skyblue')
        plt.bar(x + width/2, metrics['without_clustering']['network_lifetime'],
                width, label='Without Clustering', color='lightcoral')

        plt.xlabel('Number of Sensors')
        plt.ylabel('Network Lifetime (rounds)')
        plt.title('Network Lifetime Comparison')
        plt.xticks(x, sensor_counts)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_node_survival(sensor_counts, metrics, rounds, save_path):
        """Plot node survival rate with error handling"""
        plt.figure(figsize=(12, 6))
        rounds_range = range(rounds)

        for count_idx, count in enumerate(sensor_counts):
            if (len(metrics['with_clustering']['alive_nodes_history']) > count_idx and
                    len(metrics['with_clustering']['alive_nodes_history'][count_idx]) > 0):
                alive_history = metrics['with_clustering']['alive_nodes_history'][count_idx]
                survival_rate = [n/count * 100 for n in alive_history]
                plt.plot(rounds_range, survival_rate,
                         label=f'{count} Sensors', linestyle='--')

        plt.xlabel('Round Number')
        plt.ylabel('Survival Rate (%)')
        plt.title('Node Survival Rate Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
