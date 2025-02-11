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

## Quick Test (10-20 sensors)
### Run with clustering enabled and no clustering enable and after run validate method using the shell script
```bash
./run_all.sh
```
### Shell script -> The following script (`run_all.sh`) automates the simulation process:
```
#!/bin/bash

# Config: use "default", "test0", "test1", etc.
CONFIG_SET="test01"

# Clean up any existing results for this config
rm -rf "results/$CONFIG_SET"

# Run simulation and validation
python main.py --config config.json --set $CONFIG_SET --clustering --mode both

# View results (all under config directory)
echo "Checking results in results/$CONFIG_SET/..."
cat "results/$CONFIG_SET/network/srp_output.txt"
cat "results/$CONFIG_SET/evaluation/metrics/with_clustering_energy_consumption.csv"
cat "results/$CONFIG_SET/evaluation/metrics/without_clustering_energy_consumption.csv"
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

