import numpy as np
try:
    import cupy as cp
    try:
        GPU_INFO = cp.cuda.runtime.getDeviceProperties(0)
        GPU_NAME = GPU_INFO["name"].decode('utf-8')
        GPU_AVAILABLE = True
        print(f"\n[🚀 GPU] Found {GPU_NAME}")
    except Exception as e:
        GPU_AVAILABLE = False
        GPU_NAME = None
        print(f"\n[❌ GPU] Error initializing GPU: {str(e)}")
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None
    print("\n[❌ GPU] CUDA/cupy not installed. Install with: pip install cupy-cuda11x")
except Exception as e:
    GPU_AVAILABLE = False
    GPU_NAME = None
    print(f"\n[❌ GPU] Error initializing GPU: {str(e)}")

import numba
numba.config.THREADING_LAYER = 'tbb'


class GPUAccelerator:
    @staticmethod
    def is_available():
        if GPU_AVAILABLE:
            print(f"[🚀 GPU] Using {GPU_NAME} for computations")
            memory_info = cp.cuda.runtime.memGetInfo()
            free_memory = memory_info[0] / 1024**3  # Convert to GB
            total_memory = memory_info[1] / 1024**3
            print(
                f"[🚀 GPU] Memory: {free_memory:.2f}GB free / {total_memory:.2f}GB total")
        else:
            print("[💻 CPU] No GPU available, using CPU for all computations")
        return GPU_AVAILABLE

    @staticmethod
    def calculate_energy_matrix(sensors, relays, bandwidth, e_radio, t_amplifier):
        """GPU-accelerated energy calculation"""
        if not GPU_AVAILABLE:
            print("[💻 CPU] Falling back to CPU for energy calculations")
            return None

        try:
            print(
                f"[🚀 GPU] Processing energy matrix ({len(sensors)}x{len(relays)})")
            start_time = cp.cuda.Event()
            end_time = cp.cuda.Event()
            start_time.record()

            # Move data to GPU
            sensors_gpu = cp.asarray(sensors)
            relays_gpu = cp.asarray(relays)

            # Vectorized distance calculation
            sensor_x = sensors_gpu[:, 0].reshape(-1, 1)
            sensor_y = sensors_gpu[:, 1].reshape(-1, 1)
            relay_x = relays_gpu[:, 0].reshape(1, -1)
            relay_y = relays_gpu[:, 1].reshape(1, -1)

            distances = cp.sqrt(
                (sensor_x - relay_x)**2 +
                (sensor_y - relay_y)**2
            )

            # Calculate energy
            energy_matrix = bandwidth * \
                (e_radio + (t_amplifier * (distances ** 2)))

            # Synchronize and measure time
            end_time.record()
            end_time.synchronize()
            elapsed_time = cp.cuda.get_elapsed_time(start_time, end_time)

            print(f"[🚀 GPU] Calculation completed in {elapsed_time:.2f}ms")

            # Return to CPU
            result = cp.asnumpy(energy_matrix)
            return result

        except Exception as e:
            print(f"[⚠️ GPU Error] {str(e)}")
            print("[💻 CPU] Falling back to CPU calculation")
            return None
