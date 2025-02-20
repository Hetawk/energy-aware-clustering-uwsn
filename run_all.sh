#!/bin/bash

# Config: use "default", "test0", "test1", etc.
CONFIG_SET="test0"

# Clean up any existing results for this config
rm -rf "results/$CONFIG_SET"

# Run simulation and validation
python main.py --config config.json --set $CONFIG_SET --clustering --mode both

# View results (all under config directory)
echo "Checking results in results/$CONFIG_SET/..."
cat "results/$CONFIG_SET/network/srp_output.txt"
cat "results/$CONFIG_SET/evaluation/metrics/with_clustering_energy_consumption.csv"
cat "results/$CONFIG_SET/evaluation/metrics/without_clustering_energy_consumption.csv"