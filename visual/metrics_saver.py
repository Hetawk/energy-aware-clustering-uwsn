import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class MetricsSaver:
    @staticmethod
    def save_metrics(metrics, sensor_counts, save_dir):
        """Save metrics to CSV files"""
        metrics_dir = os.path.join(save_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # Core metrics to save
        core_metrics = [
            'network_lifetime',
            'energy_consumption',
            'alive_sensors',
            'avg_load_factor'
        ]

        for method in ['with_clustering', 'without_clustering']:
            for metric_name in core_metrics:
                if metric_name in metrics[method]:
                    filename = os.path.join(
                        metrics_dir, f"{method}_{metric_name}.csv")
                    try:
                        # Create data array with sensor counts
                        data = np.column_stack((
                            sensor_counts,
                            metrics[method][metric_name]
                        ))
                        # Save with header
                        np.savetxt(
                            filename,
                            data,
                            delimiter=',',
                            header='sensor_count,' + metric_name,
                            comments='',
                            fmt=['%d', '%.6f']
                        )
                        logging.info(
                            f"Successfully saved {metric_name} for {method} to {filename}")
                    except Exception as e:
                        logging.error(
                            f"Could not save {metric_name} for {method}: {str(e)}", exc_info=True)

        # Save clustering-specific metrics
        if 'with_clustering' in metrics and 'cluster_balance' in metrics['with_clustering']:
            filename = os.path.join(metrics_dir, "clustering_balance.csv")
            try:
                data = np.column_stack((
                    sensor_counts,
                    metrics['with_clustering']['cluster_balance']
                ))
                np.savetxt(
                    filename,
                    data,
                    delimiter=',',
                    header='sensor_count,cluster_balance',
                    comments='',
                    fmt=['%d', '%.6f']
                )
                logging.info(
                    f"Successfully saved cluster balance metrics to {filename}")
            except Exception as e:
                logging.error(
                    f"Could not save cluster balance metrics: {str(e)}", exc_info=True)

    @staticmethod
    def save_summary(metrics, save_path):
        """Save comprehensive summary to text file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            with open(save_path, 'w') as f:
                f.write("Network Performance Analysis\n")
                f.write("==========================\n\n")

                # Network Lifetime
                f.write("Network Lifetime:\n")
                f.write(
                    f"With Clustering: {np.mean(metrics['with_clustering']['network_lifetime']):.2f} rounds\n")
                f.write(
                    f"Without Clustering: {np.mean(metrics['without_clustering']['network_lifetime']):.2f} rounds\n\n")

                # Energy Efficiency
                f.write("Energy Consumption:\n")
                f.write(
                    f"With Clustering: {np.mean(metrics['with_clustering']['energy_consumption']):.2f} units\n")
                f.write(
                    f"Without Clustering: {np.mean(metrics['without_clustering']['energy_consumption']):.2f} units\n\n")

                # Load Balance
                f.write("Load Balance:\n")
                f.write(
                    f"With Clustering: {np.mean(metrics['with_clustering']['avg_load_factor']):.4f}\n")
                f.write(
                    f"Without Clustering: {np.mean(metrics['without_clustering']['avg_load_factor']):.4f}\n\n")

                # Cluster Analysis (only for clustering enabled)
                if np.any(metrics['with_clustering']['cluster_balance'] > 0):
                    f.write("Cluster Balance:\n")
                    f.write(
                        f"Average: {np.mean(metrics['with_clustering']['cluster_balance']):.4f}\n")
                    f.write(
                        f"Std Dev: {np.std(metrics['with_clustering']['cluster_balance']):.4f}\n")
            logging.info(f"Successfully saved summary to {save_path}")
        except Exception as e:
            logging.error(
                f"Could not save summary to {save_path}: {str(e)}", exc_info=True)
