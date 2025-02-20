import os
import numpy as np


def save_partial_results(result_dir, completed_sensors, relays, energies, times):
    """Save results for completed simulations"""
    try:
        os.makedirs(result_dir, exist_ok=True)

        if not completed_sensors:
            return

        # Save completed results
        with open(os.path.join(result_dir, "partial_results.txt"), 'w') as f:
            f.write("Partial Results (Interrupted Run)\n")
            f.write("==============================\n\n")
            f.write(f"Completed simulations: {len(completed_sensors)}\n\n")

            for i, count in enumerate(completed_sensors):
                f.write(f"Sensors: {count}\n")
                if i < len(relays):
                    f.write(f"Relays: {relays[i]}\n")
                    f.write(f"Energy: {energies[i]:.6f}\n")
                    f.write(f"Time: {times[i]:.2f}s\n")
                f.write("-" * 30 + "\n")

        # Also save as CSV
        save_path = os.path.join(result_dir, "partial_results.csv")
        with open(save_path, 'w') as f:
            f.write("sensors,relays,energy,time\n")
            for i, count in enumerate(completed_sensors):
                if i < len(relays):
                    f.write(
                        f"{count},{relays[i]},{energies[i]:.6f},{times[i]:.2f}\n")

    except Exception as e:
        print(f"[❌ Error] Failed to save partial results: {str(e)}")
