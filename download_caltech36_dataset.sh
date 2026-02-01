#!/bin/bash
# Script to download the Caltech36 Facebook social network dataset

echo "Downloading Caltech36 Facebook social network dataset..."
echo "Source: https://networkrepository.com/socfb-Caltech36.php"

# Create data directory if it doesn't exist
mkdir -p data

# Download the dataset (Network Repository provides zip files)
cd data

# Try to download the zip file
if command -v wget &> /dev/null; then
    echo "Downloading using wget..."
    wget https://nrvis.com/download/data/social/socfb-Caltech36.zip
elif command -v curl &> /dev/null; then
    echo "Downloading using curl..."
    curl -O https://nrvis.com/download/data/social/socfb-Caltech36.zip
else
    echo "Error: Neither wget nor curl is available."
    echo "Please download manually from: https://networkrepository.com/socfb-Caltech36.php"
    exit 1
fi

# Extract the zip file
if command -v unzip &> /dev/null; then
    echo "Extracting zip file..."
    unzip -o socfb-Caltech36.zip
    echo "Extraction complete!"
    
    # Find the edge list file
    if [ -f "socfb-Caltech36.edges" ]; then
        echo "Found: socfb-Caltech36.edges"
    elif [ -f "socfb-Caltech36.txt" ]; then
        echo "Found: socfb-Caltech36.txt"
    else
        echo "Warning: Edge list file not found. Please check the extracted files."
        ls -la
    fi
else
    echo "Warning: unzip is not available. Please extract manually."
    echo "The edge list file should be named: socfb-Caltech36.edges or socfb-Caltech36.txt"
fi

echo ""
echo "Download complete!"
echo "File location: data/socfb-Caltech36.edges (or .txt)"
echo ""
echo "You can now run: python -m src.caltech36_experiments"
