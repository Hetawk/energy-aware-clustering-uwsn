import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np


class ParallelProcessor:
    @staticmethod
    def batch_process(data, func, batch_size=1000, max_workers=None):
        """Process data in batches using parallel processing"""
        if max_workers is None:
            max_workers = mp.cpu_count()

        total_items = len(data)
        num_batches = (total_items + batch_size - 1) // batch_size

        print(f"\n[💻 CPU] Processing {total_items} items")
        print(f"[💻 CPU] Using {max_workers} CPU cores")
        print(
            f"[💻 CPU] Split into {num_batches} batches of {batch_size} items")

        # Split data into batches
        batches = [data[i:i + batch_size]
                   for i in range(0, len(data), batch_size)]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(func, batches))

        print(f"[💻 CPU] Parallel processing completed")
        return np.concatenate(results) if isinstance(results[0], np.ndarray) else results

    @staticmethod
    def parallel_matrix_ops(matrix_ops, max_workers=None):
        """Execute matrix operations in parallel"""
        if max_workers is None:
            max_workers = mp.cpu_count()
        print(
            f"[💻 CPU] Running matrix operations across {max_workers} threads")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda f: f(), matrix_ops))
