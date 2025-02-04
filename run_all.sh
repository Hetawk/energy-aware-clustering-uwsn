#!/bin/bash

# Run with clustering enabled
python main.py --config config.json --set default --clustering

# Run without clustering for comparison
python main.py --config config.json --set default --no-clustering

# Validate implementation
python validate_implementation.py

# View sleep scheduling results
cat results/network/srp_output.txt

# Check energy consumption with clustering
cat results/evaluation/metrics/with_clustering_energy_consumption.csv

# Check energy consumption without clustering
cat results/evaluation/metrics/without_clustering_energy_consumption.csv