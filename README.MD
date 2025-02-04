# Energy-Aware Clustering and Sleep Scheduling Protocol for UWSNs

An implementation of an Energy-Aware Clustering and Sleep Scheduling Routing Protocol designed to extend network lifespan in Underwater Wireless Sensor Networks (UWSNs).

## Key Features

- **Fuzzy C-Means Clustering**: Energy-efficient sensor grouping
- **Energy-aware Cluster Head Selection**: Optimal selection based on energy levels
- **Adaptive Sleep Scheduling**: Dynamic scheduling using membership values
- **Load Balancing**: Extended network lifetime through balanced energy consumption
- **Oil Spill Detection**: Environmental monitoring capability

## Installation & Dependencies

```bash
# Install required packages
pip install numpy matplotlib scikit-fuzzy pulp networkx
```

## Project Structure
.
├── config.json            # Configuration settings
├── main.py               # Main execution script
├── optimization.py       # Clustering and optimization
├── simulation.py         # Network simulation
├── energy_calculation.py # Energy calculations
└── results/             # Simulation outputs
    ├── network/         # Network statistics
    ├── energy/          # Energy consumption data
    ├── state/          # Sensor states
    └── evaluation/      # Performance metrics

## Quick Test (10-20 sensors)
### Run with clustering enabled
```bash
python main.py --config config.json --set test0 --clustering
```

### Run without clustering for comparison
```bash
python main.py --config config.json --set test0 --no-clustering
```

### Validate implementation
```bash
python validate_implementation.py
```

## Check Results
### View sleep scheduling results
```bash
cat results/network/srp_output.txt
```

### Check energy consumption
```bash
cat results/evaluation/metrics/with_clustering_energy_consumption.csv
cat results/evaluation/metrics/without_clustering_energy_consumption.csv
```

## Configuration Options
Edit config.json to modify simulation parameters:
```json
{
  "configurations": {
    "default": {
      "radius": 30,
      "sensor_counts": [100, 200, 300, 400, 500],
      "rounds": 10,
      "width": 50.0,
      "relay_constraint": 50
    },
    "test0": {
      "radius": 10,
      "sensor_counts": [10, 15, 20],  
      "rounds": 5, 
      "width": 20.0,
      "relay_constraint": 10
    },
    "test1": {
      "radius": 20,
      "sensor_counts": [50, 100, 150],
      "rounds": 5,
      "width": 30.0,
      "relay_constraint": 30
    },
    "test2": {
      "radius": 40,
      "sensor_counts": [200, 300, 400],
      "rounds": 15,
      "width": 70.0,
      "relay_constraint": 70
    }
  }
}
```

