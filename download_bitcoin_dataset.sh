#!/bin/bash
# Script to download the Bitcoin OTC dataset

echo "Downloading Bitcoin OTC trust network dataset..."
echo "Source: https://snap.stanford.edu/data/soc-sign-bitcoinotc.html"

# Create data directory if it doesn't exist
mkdir -p data

# Download the dataset
cd data
wget https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz

echo "Download complete!"
echo "File location: data/soc-sign-bitcoinotc.csv.gz"
echo ""
echo "You can now run: python -m src.bitcoin_experiments"
