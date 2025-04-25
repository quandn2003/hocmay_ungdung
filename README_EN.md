# Time Series Stock Clustering

This project implements clustering of time series stock data using K-means algorithm with AI-Daoud initialization method, comparing it with traditional random initialization.

## Overview

The aim of this project is to cluster companies based on the similarity of their stock price fluctuations over time. The implementation uses an advanced initialization method (AI-Daoud) for K-means clustering and compares it with traditional random initialization.

## Features

- Advanced time series feature extraction (time domain, frequency domain, statistical)
- K-means clustering with AI-Daoud initialization
- Comparison with traditional random initialization
- Automatic detection of optimal cluster number using the Elbow Method
- Evaluation using multiple metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, SSE/WCSS
- 2D and 3D visualization of clusters using PCA

## Data Format

The dataset consists of time series data of stock prices stored in .dat files in the `./data` directory. Each file represents a company and contains stock prices collected at regular intervals.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/time-series-stock-clustering.git
   cd time-series-stock-clustering
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main clustering script:
```
python kmeans_clustering.py
```

By default, the script will test k values of 3, 5, 7, and 10. You can specify custom cluster values:

```
# Test with 4, 6, and 8 clusters
python kmeans_clustering.py --clusters 4 6 8

# Test with a single cluster value
python kmeans_clustering.py -k 5
```

To use the Elbow Method to automatically find the optimal number of clusters:

```
# Run the Elbow Method with default range (2-15)
python kmeans_clustering.py --elbow

# Run the Elbow Method with custom range
python kmeans_clustering.py --elbow --min-k 2 --max-k 20 --step-k 2
```

## Output

The script generates:
- Terminal output with metrics and cluster distribution
- 2D and 3D visualization of clusters
- Comparison plots of evaluation metrics
- Elbow method plot (when using the --elbow option)

## Implementation Details

### AI-Daoud Initialization Algorithm

The AI-Daoud initialization for K-means works as follows:
1. Calculate variance of each feature attribute
2. Find the attribute with the highest variance
3. Sort the data points by this feature
4. Divide the sorted data into k equal-sized groups
5. Find the median of each group
6. Use the data points corresponding to these medians as initial centroids

### Feature Extraction

The implementation extracts various features from time series data:
- Time domain features (statistical measures, volatility, etc.)
- Frequency domain features (FFT-based, spectral analysis)
- Statistical features (autocorrelation, trend analysis, etc.)

## Files

- `kmeans_clustering.py`: Main clustering implementation
- `time_series_features.py`: Advanced feature extraction for time series data
- `BaoCao.md`: Detailed report in Vietnamese
- `README_EN.md`: English documentation
- `data/`: Directory containing time series data in .dat files

## License

MIT

## Author

Your Name 