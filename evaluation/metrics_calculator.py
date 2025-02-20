import numpy as np


class MetricsCalculator:
    @staticmethod
    def calculate_network_lifetime(final_energy):
        """Calculate network lifetime as rounds until first node dies"""
        try:
            final_energy = np.array(final_energy) if not isinstance(
                final_energy, np.ndarray) else final_energy
            node_energies = np.sum(final_energy, axis=1)
            return float(np.min(node_energies))
        except Exception as e:
            print(f"Network lifetime calculation error: {e}")
            return 0.0

    @staticmethod
    def calculate_energy_consumption(initial_energy, final_energy):
        """Calculate total energy consumed"""
        return np.sum(initial_energy) - np.sum(final_energy)

    @staticmethod
    def calculate_alive_sensors(final_energy):
        """Calculate ratio of alive sensors"""
        total = len(final_energy)
        alive = sum(1 for row in final_energy if np.sum(row) > 0)
        return alive / total

    @staticmethod
    def calculate_cluster_balance(cluster_labels, final_energy):
        """Calculate energy balance among clusters"""
        unique_clusters = np.unique(cluster_labels)
        cluster_energies = []

        for cluster in unique_clusters:
            cluster_nodes = np.where(cluster_labels == cluster)[0]
            cluster_energy = np.mean([np.sum(final_energy[i])
                                     for i in cluster_nodes])
            cluster_energies.append(cluster_energy)

        return np.array(cluster_energies)
