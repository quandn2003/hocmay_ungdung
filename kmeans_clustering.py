import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from dtw import dtw
from numpy.linalg import norm
from time_series_features import extract_all_features
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import argparse
from kneed import KneeLocator
warnings.filterwarnings('ignore')

def load_data(data_dir='./data'):
    """
    Load time series data from .dat files
    """
    files = glob.glob(os.path.join(data_dir, "*.dat"))
    data = {}
    
    for file_path in files:
        company = os.path.basename(file_path).replace(".dat", "")
        with open(file_path, 'r') as f:
            # Skip empty lines
            time_series = np.array([float(line.strip()) for line in f if line.strip()])
        data[company] = time_series
        
    return data

def preprocess_data(data):
    """
    Preprocess time series data
    - Z-score normalization
    - Ensure all series have the same length
    """
    # Find minimum length
    min_length = min(len(ts) for ts in data.values())
    
    # Normalize and ensure same length
    preprocessed_data = {}
    for company, time_series in data.items():
        # Truncate to minimum length
        time_series = time_series[:min_length]
        
        # Z-score normalization
        mean = np.mean(time_series)
        std = np.std(time_series)
        if std != 0:
            normalized_ts = (time_series - mean) / std
        else:
            normalized_ts = time_series - mean
            
        preprocessed_data[company] = normalized_ts
        
    return preprocessed_data

def extract_features(data, use_advanced_features=True):
    """
    Extract features from time series data
    """
    features = {}
    companies = []
    feature_names = None
    
    for company, time_series in data.items():
        companies.append(company)
        
        if use_advanced_features:
            # Use advanced feature extraction
            feature_values, feature_names_list = extract_all_features(time_series)
            if feature_names is None:
                feature_names = feature_names_list
            features[company] = feature_values
        else:
            # Simple feature extraction
            mean = np.mean(time_series)
            std = np.std(time_series)
            min_val = np.min(time_series)
            max_val = np.max(time_series)
            range_val = max_val - min_val
            
            # Trend features
            n = len(time_series)
            x = np.arange(n)
            slope, _ = np.polyfit(x, time_series, 1)
            
            # Volatility features
            returns = np.diff(time_series) / time_series[:-1]
            volatility = np.std(returns)
            
            # Statistical features
            skewness = np.mean(((time_series - mean) / std) ** 3) if std != 0 else 0
            kurtosis = np.mean(((time_series - mean) / std) ** 4) if std != 0 else 0
            
            # Combine features
            feature_vector = np.array([
                mean, std, min_val, max_val, range_val, 
                slope, volatility, skewness, kurtosis
            ])
            
            features[company] = feature_vector
    
    # Convert to numpy array
    X = np.array([features[company] for company in companies])
    
    # Handle NaN values
    X = np.nan_to_num(X)
    
    return X, companies, feature_names

def apply_pca(X, n_components=0.95):
    """
    Apply PCA to reduce dimensionality while retaining specified variance
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Reduced dimensions from {X.shape[1]} to {X_pca.shape[1]} features")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return X_pca, pca

def ai_daoud_initialization(X, k):
    """
    AI-Daoud initialization method for K-means
    """
    n_samples, n_features = X.shape
    
    # Step 1: Calculate variance of each feature
    variances = np.var(X, axis=0)
    
    # Step 2: Find the feature with highest variance
    max_var_feature_idx = np.argmax(variances)
    
    # Step 3: Sort the data points by this feature
    feature_values = X[:, max_var_feature_idx]
    sorted_indices = np.argsort(feature_values)
    
    # Step 4: Divide into k equal sized groups
    groups = np.array_split(sorted_indices, k)
    
    # Step 5: Find the median of each group and use corresponding data point as centroid
    centroids = np.zeros((k, n_features))
    for i, group in enumerate(groups):
        if len(group) > 0:
            median_idx = group[len(group) // 2]
            centroids[i] = X[median_idx]
    
    return centroids

def kmeans(X, k, max_iters=100, init_method='ai_daoud'):
    """
    K-means clustering algorithm
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training instances to cluster
    k : int
        Number of clusters to form
    max_iters : int, default=100
        Maximum number of iterations
    init_method : str, default='ai_daoud'
        Centroid initialization method ('ai_daoud' or 'random')
    
    Returns:
    --------
    centroids : array, shape (k, n_features)
        Coordinates of cluster centers
    labels : array, shape (n_samples,)
        Labels of each point
    inertia : float
        Sum of squared distances to closest centroid
    """
    n_samples, n_features = X.shape
    
    # Initialize centroids based on chosen method
    if init_method == 'ai_daoud':
        centroids = ai_daoud_initialization(X, k)
    else:  # random initialization
        random_indices = np.random.choice(n_samples, k, replace=False)
        centroids = X[random_indices]
    
    # Initialize labels
    labels = np.zeros(n_samples, dtype=int)
    old_labels = np.ones(n_samples, dtype=int)
    
    # Main loop
    iteration = 0
    while not np.array_equal(labels, old_labels) and iteration < max_iters:
        old_labels = labels.copy()
        
        # Assign samples to closest centroids
        distances = cdist(X, centroids, 'euclidean')
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        for j in range(k):
            if np.sum(labels == j) > 0:  # Ensure the cluster is not empty
                centroids[j] = np.mean(X[labels == j], axis=0)
        
        iteration += 1
    
    print(f"K-means converged after {iteration} iterations")
    
    # Calculate inertia (sum of squared distances to assigned centroid)
    inertia = 0
    for i in range(n_samples):
        inertia += np.sum((X[i] - centroids[labels[i]]) ** 2)
    
    return centroids, labels, inertia

def silhouette_score(X, labels, k):
    """
    Calculate the silhouette score
    """
    n_samples = X.shape[0]
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Cluster of current sample
        cluster_i = labels[i]
        
        # Calculate a (mean distance to all other samples in the same cluster)
        if np.sum(labels == cluster_i) > 1:  # If not alone in cluster
            a = np.mean(cdist(X[i].reshape(1, -1), X[labels == cluster_i], 'euclidean'))
        else:
            a = 0
        
        # Calculate b (mean distance to samples in nearest neighboring cluster)
        b_values = []
        for j in range(k):
            if j != cluster_i and np.sum(labels == j) > 0:
                b_values.append(np.mean(cdist(X[i].reshape(1, -1), X[labels == j], 'euclidean')))
        
        b = min(b_values) if b_values else 0
        
        # Calculate silhouette
        if a == 0 and b == 0:
            silhouette_vals[i] = 0
        elif a < b:
            silhouette_vals[i] = 1 - a / b
        elif a > b:
            silhouette_vals[i] = b / a - 1
        else:  # a == b
            silhouette_vals[i] = 0
    
    # Mean silhouette value
    return np.mean(silhouette_vals)

def davies_bouldin_index(X, labels, centroids, k):
    """
    Calculate the Davies-Bouldin Index
    """
    if k <= 1:
        return 0
    
    # Calculate cluster dispersions (average distance of all samples to centroid)
    dispersions = np.zeros(k)
    for i in range(k):
        cluster_samples = X[labels == i]
        if len(cluster_samples) > 0:
            distances = cdist(cluster_samples, centroids[i].reshape(1, -1), 'euclidean')
            dispersions[i] = np.mean(distances)
    
    # Calculate Davies-Bouldin Index
    db_indices = np.zeros(k)
    for i in range(k):
        if np.sum(labels == i) == 0:  # Skip empty cluster
            continue
            
        max_ratio = 0
        for j in range(k):
            if i != j and np.sum(labels == j) > 0:
                # Distance between centroids
                centroid_distance = np.linalg.norm(centroids[i] - centroids[j])
                if centroid_distance > 0:  # Avoid division by zero
                    ratio = (dispersions[i] + dispersions[j]) / centroid_distance
                    max_ratio = max(max_ratio, ratio)
        
        db_indices[i] = max_ratio
    
    # Average over all clusters
    return np.mean(db_indices)

def calinski_harabasz_index(X, labels, centroids, k):
    """
    Calculate the Calinski-Harabasz Index
    """
    n_samples = X.shape[0]
    
    if k <= 1 or n_samples <= k:
        return 0
    
    # Calculate overall centroid
    overall_centroid = np.mean(X, axis=0)
    
    # Between-cluster dispersion (weighted sum of distances between cluster centroids and overall centroid)
    between_cluster_ss = 0
    for i in range(k):
        n_cluster_samples = np.sum(labels == i)
        if n_cluster_samples > 0:
            between_cluster_ss += n_cluster_samples * np.sum((centroids[i] - overall_centroid) ** 2)
    
    # Within-cluster dispersion
    within_cluster_ss = 0
    for i in range(n_samples):
        within_cluster_ss += np.sum((X[i] - centroids[labels[i]]) ** 2)
    
    # Calculate CH index
    if within_cluster_ss == 0:  # Perfect clustering
        return float('inf')
    
    ch_index = (between_cluster_ss / (k - 1)) / (within_cluster_ss / (n_samples - k))
    return ch_index

def find_most_influential_features(X, feature_names=None):
    """
    Find the most influential features based on the covariance matrix
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data matrix
    feature_names : list or None
        Names of features corresponding to columns in X
        
    Returns:
    --------
    influential_indices : list
        Indices of the most influential features
    influential_names : list
        Names of the most influential features (if feature_names provided)
    """
    # Calculate the covariance matrix
    cov_matrix = np.cov(X, rowvar=False)
    
    # Get the diagonal elements (variances)
    variances = np.diag(cov_matrix)
    
    # Find indices of features with highest variance
    influential_indices = np.argsort(variances)[::-1][:3]
    
    # Get feature names if provided
    influential_names = None
    if feature_names is not None:
        influential_names = [feature_names[i] for i in influential_indices]
    
    return influential_indices, influential_names

def find_most_influential_pca_components(pca):
    """
    Find the three most influential PCA components based on explained variance ratio
    
    Parameters:
    -----------
    pca : PCA object
        Fitted PCA object with explained_variance_ratio_ attribute
        
    Returns:
    --------
    influential_indices : list
        Indices of the most influential PCA components
    """
    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Find indices of PCA components with highest variance explanation
    influential_indices = np.argsort(explained_variance_ratio)[::-1][:3]
    
    return influential_indices

def visualize_clusters_3d(X_pca, labels, companies, title='Cluster Visualization 3D', pca=None):
    """
    Visualize clusters in 3D space using the three most influential PCA components
    """
    # Create result directory if it doesn't exist
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Created directory: {result_dir}")
    
    # If PCA object is provided, use most influential components
    if pca is not None and hasattr(pca, 'explained_variance_ratio_'):
        influential_indices = find_most_influential_pca_components(pca)
        X_3d = X_pca[:, influential_indices]
        
        # Update title and axis labels
        component_indices = [i+1 for i in influential_indices]  # 1-indexed for display
        title = f"{title} (Most Influential PCA Components)"
        xlabel = f"PCA Component {component_indices[0]} ({pca.explained_variance_ratio_[influential_indices[0]]:.2%})"
        ylabel = f"PCA Component {component_indices[1]} ({pca.explained_variance_ratio_[influential_indices[1]]:.2%})"
        zlabel = f"PCA Component {component_indices[2]} ({pca.explained_variance_ratio_[influential_indices[2]]:.2%})"
    else:
        # Fallback to first three components
        X_3d = X_pca[:, :3]
        title = f"{title} (First 3 PCA Components)"
        xlabel = "Principal Component 1"
        ylabel = "Principal Component 2"
        zlabel = "Principal Component 3"
    
    # Get unique clusters
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    
    # Create color map
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, cluster in enumerate(unique_clusters):
        cluster_points = X_3d[labels == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                   c=[cmap(i)], label=f'Cluster {cluster+1}', alpha=0.7)
    
    # Add labels for some points (showing company names)
    for i, (company, x, y, z) in enumerate(zip(companies, X_3d[:, 0], X_3d[:, 1], X_3d[:, 2])):
        if i % 20 == 0:  # Label every 20th point to avoid clutter in 3D
            ax.text(x, y, z, company, fontsize=8)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()
    plt.tight_layout()
    
    # Save figure to result directory
    filename = os.path.join(result_dir, f'{title.replace(" ", "_").lower()}.png')
    plt.savefig(filename)
    plt.close()
    
    print(f"3D cluster visualization saved as '{filename}'")

def visualize_clusters(X_pca, labels, companies, title='Cluster Visualization', pca=None):
    """
    Visualize clusters in 3D space using the most influential PCA components
    """
    # Skip 2D visualization and only do 3D visualization if we have at least 3 components
    if X_pca.shape[1] >= 3:
        visualize_clusters_3d(X_pca, labels, companies, title, pca)
    else:
        print(f"Warning: Cannot create 3D visualization, only {X_pca.shape[1]} components available")

def find_elbow_point(k_values, sse_values):
    """
    Find the elbow point in the SSE curve using the Kneed package
    
    Parameters:
    -----------
    k_values : array
        Array of k values tested
    sse_values : array
        Array of SSE values corresponding to each k
        
    Returns:
    --------
    optimal_k : int
        The optimal k value based on the elbow method
    """
    # Ensure k_values are sorted in ascending order
    sorted_idx = np.argsort(k_values)
    k_values_sorted = np.array(k_values)[sorted_idx]
    sse_values_sorted = np.array(sse_values)[sorted_idx]
    
    # Find the elbow point using KneeLocator
    try:
        kneedle = KneeLocator(
            k_values_sorted, 
            sse_values_sorted, 
            curve='convex', 
            direction='decreasing',
            S=1.0
        )
        elbow_k = kneedle.elbow
        
        # If no elbow is found, use the k with highest curvature
        if elbow_k is None:
            elbow_k = k_values_sorted[np.argmax(np.diff(np.diff(sse_values_sorted)))]
        
        # Fall back to the middle value if still no elbow
        if elbow_k is None:
            elbow_k = k_values_sorted[len(k_values_sorted) // 2]
            
        return int(elbow_k)
    except:
        # If the kneed package fails, use a simple heuristic approach
        # Calculate the rate of SSE decrease
        sse_diffs = np.diff(sse_values_sorted)
        rates = sse_diffs[:-1] / sse_diffs[1:]
        
        # Find the point with maximum change in rate
        if len(rates) > 0:
            elbow_idx = np.argmax(rates) + 1
            return int(k_values_sorted[elbow_idx])
        else:
            return int(k_values_sorted[len(k_values_sorted) // 2])

def plot_elbow_method(k_values, sse_values, optimal_k, method='AI-Daoud'):
    """
    Plot the Elbow Method curve and mark the optimal k
    
    Parameters:
    -----------
    k_values : array
        Array of k values tested
    sse_values : array
        Array of SSE values corresponding to each k
    optimal_k : int
        The optimal k value to mark on the plot
    method : str
        Initialization method name for the title
    """
    # Create result directory if it doesn't exist
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sse_values, 'bo-')
    plt.plot(optimal_k, sse_values[k_values.index(optimal_k)], 'ro', markersize=10, 
             label=f'Elbow Point: k={optimal_k}')
    
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title(f'Elbow Method for {method} K-means')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure to result directory
    filename = os.path.join(result_dir, f'elbow_method_{method.lower().replace("-", "_")}.png')
    plt.savefig(filename)
    plt.close()
    
    print(f"Elbow method plot saved as '{filename}'")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='K-means clustering for time series stock data')
    parser.add_argument('--clusters', '-k', type=int, nargs='+', default=[3, 5, 7, 10],
                        help='Number of clusters to try (can specify multiple values)')
    parser.add_argument('--elbow', '-e', action='store_true',
                        help='Use the Elbow Method to find optimal k')
    parser.add_argument('--min-k', type=int, default=2,
                        help='Minimum number of clusters for Elbow Method range (default: 2)')
    parser.add_argument('--max-k', type=int, default=15,
                        help='Maximum number of clusters for Elbow Method range (default: 15)')
    parser.add_argument('--step-k', type=int, default=1,
                        help='Step size for k values in Elbow Method range (default: 1)')
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} time series.")
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessed_data = preprocess_data(data)
    
    # Extract features
    print("Extracting advanced time series features...")
    X, companies, feature_names = extract_features(preprocessed_data, use_advanced_features=True)
    print(f"Feature matrix shape: {X.shape}")
    
    # Apply PCA for dimensionality reduction
    print("Applying PCA for dimensionality reduction...")
    X_pca, pca = apply_pca(X, n_components=0.95)
    
    # Determine k values to try
    if args.elbow:
        print("\nRunning Elbow Method to find optimal k...")
        
        # Use range from min-k to max-k with step-k
        k_values = list(range(args.min_k, args.max_k + 1, args.step_k))
        print(f"Testing k values from {args.min_k} to {args.max_k} with step {args.step_k}")
        
        # Run K-means with AI-Daoud for each k and collect SSE values
        ai_daoud_sse_values = []
        for k in k_values:
            print(f"Running K-means with AI-Daoud initialization for k = {k}...")
            _, _, inertia = kmeans(X_pca, k, init_method='ai_daoud')
            ai_daoud_sse_values.append(inertia)
        
        # Find optimal k using the Elbow Method
        optimal_k = find_elbow_point(k_values, ai_daoud_sse_values)
        print(f"\nElbow Method suggests optimal k = {optimal_k} for AI-Daoud initialization")
        
        # Plot the Elbow Method curve
        plot_elbow_method(k_values, ai_daoud_sse_values, optimal_k, method="AI-Daoud")
        
        # Set the optimal k as the only k to use for detailed analysis
        k_values = [optimal_k]
    else:
        # Use the specified clusters
        k_values = args.clusters
    
    print(f"\nPerforming detailed analysis for cluster values: {k_values}")
    
    results = []
    
    for k in k_values:
        print(f"\nRunning for k = {k}")
        
        # AI-Daoud K-means
        print("Running K-means with AI-Daoud initialization...")
        ai_daoud_centroids, ai_daoud_labels, ai_daoud_inertia = kmeans(X_pca, k, init_method='ai_daoud')
        
        # Traditional K-means
        print("Running K-means with random initialization...")
        random_centroids, random_labels, random_inertia = kmeans(X_pca, k, init_method='random')
        
        # Calculate metrics for AI-Daoud
        ai_daoud_silhouette = silhouette_score(X_pca, ai_daoud_labels, k)
        ai_daoud_db = davies_bouldin_index(X_pca, ai_daoud_labels, ai_daoud_centroids, k)
        ai_daoud_ch = calinski_harabasz_index(X_pca, ai_daoud_labels, ai_daoud_centroids, k)
        ai_daoud_sse = ai_daoud_inertia
        
        # Calculate metrics for traditional K-means
        random_silhouette = silhouette_score(X_pca, random_labels, k)
        random_db = davies_bouldin_index(X_pca, random_labels, random_centroids, k)
        random_ch = calinski_harabasz_index(X_pca, random_labels, random_centroids, k)
        random_sse = random_inertia
        
        # Store results
        results.append({
            'k': k,
            'AI-Daoud': {
                'Silhouette': ai_daoud_silhouette,
                'Davies-Bouldin': ai_daoud_db,
                'Calinski-Harabasz': ai_daoud_ch,
                'SSE/WCSS': ai_daoud_sse,
                'labels': ai_daoud_labels,
                'centroids': ai_daoud_centroids
            },
            'Random': {
                'Silhouette': random_silhouette,
                'Davies-Bouldin': random_db,
                'Calinski-Harabasz': random_ch,
                'SSE/WCSS': random_sse,
                'labels': random_labels,
                'centroids': random_centroids
            }
        })
        
        # Print comparison
        print(f"\nComparison of metrics for k = {k}:")
        print(f"{'Metric':<20} {'AI-Daoud':<15} {'Random':<15}")
        print("-" * 50)
        print(f"{'Silhouette (↑)':<20} {ai_daoud_silhouette:<15.4f} {random_silhouette:<15.4f}")
        print(f"{'Davies-Bouldin (↓)':<20} {ai_daoud_db:<15.4f} {random_db:<15.4f}")
        print(f"{'Calinski-Harabasz (↑)':<20} {ai_daoud_ch:<15.4f} {random_ch:<15.4f}")
        print(f"{'SSE/WCSS (↓)':<20} {ai_daoud_sse:<15.4f} {random_sse:<15.4f}")
        
        # Cluster distribution
        print("\nCluster distribution:")
        for method, method_labels in [("AI-Daoud", ai_daoud_labels), ("Random", random_labels)]:
            counts = [np.sum(method_labels == i) for i in range(k)]
            print(f"{method}: {counts}")
        
        # Visualize clusters with most influential PCA components
        visualize_clusters(X_pca, ai_daoud_labels, companies, f'AI-Daoud Clusters (k={k})', pca)
        visualize_clusters(X_pca, random_labels, companies, f'Random Clusters (k={k})', pca)
    
    # If we have multiple k values, find the best k based on silhouette score
    if len(k_values) > 1:
        best_k_ai_daoud = max(results, key=lambda x: x['AI-Daoud']['Silhouette'])['k']
        best_k_random = max(results, key=lambda x: x['Random']['Silhouette'])['k']
        
        print(f"\nBest k based on Silhouette score:")
        print(f"AI-Daoud: k = {best_k_ai_daoud}")
        print(f"Random: k = {best_k_random}")
        
        # Get best clustering results
        best_result = next(r for r in results if r['k'] == best_k_ai_daoud)
        best_labels = best_result['AI-Daoud']['labels']
    else:
        # Only one k value was tested
        best_labels = results[0]['AI-Daoud']['labels']
        best_k_ai_daoud = k_values[0]
    
    # Display some cluster members
    print("\nSample cluster members (AI-Daoud):")
    for i in range(best_k_ai_daoud):
        cluster_companies = [companies[j] for j in range(len(companies)) if best_labels[j] == i]
        if cluster_companies:
            print(f"Cluster {i+1}: {', '.join(cluster_companies[:5])}{'...' if len(cluster_companies) > 5 else ''}")
    
    # Plot comparison of metrics for different k values if we have multiple k
    if len(k_values) > 1:
        plot_metrics(results)

def plot_metrics(results):
    """
    Plot metrics comparison between AI-Daoud and random initialization
    """
    # Create result directory if it doesn't exist
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    k_values = [r['k'] for r in results]
    
    # Silhouette score (higher is better)
    ai_daoud_silhouette = [r['AI-Daoud']['Silhouette'] for r in results]
    random_silhouette = [r['Random']['Silhouette'] for r in results]
    
    # Davies-Bouldin index (lower is better)
    ai_daoud_db = [r['AI-Daoud']['Davies-Bouldin'] for r in results]
    random_db = [r['Random']['Davies-Bouldin'] for r in results]
    
    # Calinski-Harabasz index (higher is better)
    ai_daoud_ch = [r['AI-Daoud']['Calinski-Harabasz'] for r in results]
    random_ch = [r['Random']['Calinski-Harabasz'] for r in results]
    
    # SSE/WCSS (lower is better)
    ai_daoud_sse = [r['AI-Daoud']['SSE/WCSS'] for r in results]
    random_sse = [r['Random']['SSE/WCSS'] for r in results]
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Silhouette score
    axs[0, 0].plot(k_values, ai_daoud_silhouette, 'o-', label='AI-Daoud')
    axs[0, 0].plot(k_values, random_silhouette, 's--', label='Random')
    axs[0, 0].set_title('Silhouette Score (higher is better)')
    axs[0, 0].set_xlabel('Number of clusters (k)')
    axs[0, 0].set_ylabel('Silhouette Score')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Davies-Bouldin index
    axs[0, 1].plot(k_values, ai_daoud_db, 'o-', label='AI-Daoud')
    axs[0, 1].plot(k_values, random_db, 's--', label='Random')
    axs[0, 1].set_title('Davies-Bouldin Index (lower is better)')
    axs[0, 1].set_xlabel('Number of clusters (k)')
    axs[0, 1].set_ylabel('Davies-Bouldin Index')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Calinski-Harabasz index
    axs[1, 0].plot(k_values, ai_daoud_ch, 'o-', label='AI-Daoud')
    axs[1, 0].plot(k_values, random_ch, 's--', label='Random')
    axs[1, 0].set_title('Calinski-Harabasz Index (higher is better)')
    axs[1, 0].set_xlabel('Number of clusters (k)')
    axs[1, 0].set_ylabel('Calinski-Harabasz Index')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # SSE/WCSS
    axs[1, 1].plot(k_values, ai_daoud_sse, 'o-', label='AI-Daoud')
    axs[1, 1].plot(k_values, random_sse, 's--', label='Random')
    axs[1, 1].set_title('SSE/WCSS (lower is better)')
    axs[1, 1].set_xlabel('Number of clusters (k)')
    axs[1, 1].set_ylabel('SSE/WCSS')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save to result directory
    filename = os.path.join(result_dir, 'metrics_comparison.png')
    plt.savefig(filename)
    plt.close()
    
    print(f"Metrics comparison plot saved as '{filename}'")

if __name__ == "__main__":
    main() 