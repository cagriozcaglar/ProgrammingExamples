import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# From https://claude.ai/chat/0d9435af-87c8-4e07-8e49-ecafdd765833
class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        """
        K-means clustering algorithm implementation.
        
        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters to form and centroids to generate.
        max_iters : int, default=100
            Maximum number of iterations for a single run.
        random_state : int, optional
            Seed for random number generation.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
    
    def _init_centroids(self, X):
        """Initialize centroids by randomly selecting points from the dataset."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Randomly select n_clusters data points as initial centroids
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[indices]
    
    def _assign_clusters(self, X):
        """Assign each data point to the nearest centroid."""
        # Calculate distances from each point to each centroid
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            # Euclidean distance calculation
            distances[:, k] = np.sqrt(np.sum((X - self.centroids[k])**2, axis=1))
        
        # Assign each point to the nearest centroid
        self.labels = np.argmin(distances, axis=1)
        
        # Calculate inertia (sum of squared distances to closest centroid)
        self.inertia_ = np.sum([np.sum((X[self.labels == k] - self.centroids[k])**2) 
                              for k in range(self.n_clusters)])
        
        return self.labels
    
    def _update_centroids(self, X):
        """Update centroids based on the mean of the assigned points."""
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.n_clusters):
            if np.sum(self.labels == k) > 0:  # Avoid empty clusters
                new_centroids[k] = np.mean(X[self.labels == k], axis=0)
            else:
                # If a cluster is empty, reinitialize that centroid
                new_centroids[k] = X[np.random.choice(X.shape[0])]
        
        # Check if centroids have converged
        centroid_change = np.sum(np.abs(new_centroids - self.centroids))
        self.centroids = new_centroids
        return centroid_change
    
    def fit(self, X):
        """
        Compute k-means clustering.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns:
        --------
        self : object
            Fitted estimator.
        """
        # Initialize centroids
        self._init_centroids(X)
        
        # Iterative optimization
        for i in range(self.max_iters):
            # Assign points to clusters
            self._assign_clusters(X)
            
            # Update centroids
            centroid_change = self._update_centroids(X)
            
            # Check for convergence
            if centroid_change < 1e-4:
                break
                
        return self
    
    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
            
        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        # Calculate distances from each point to each centroid
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sqrt(np.sum((X - self.centroids[k])**2, axis=1))
        
        # Assign each point to the nearest centroid
        return np.argmin(distances, axis=1)

# Example usage
def demo_kmeans():
    # Generate sample data
    n_samples = 1500
    X, y_true = make_blobs(n_samples=n_samples, centers=4, cluster_std=0.7, random_state=42)
    
    # Fit the K-means model
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)
    y_pred = kmeans.labels
    
    # Visualize the results
    plt.figure(figsize=(10, 8))
    
    # Plot original data colored by predicted clusters
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=30, cmap='viridis', alpha=0.7)
    
    # Plot centroids
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                c='red', s=200, alpha=0.7, marker='X',
                label='Centroids')
    
    plt.title('K-means Clustering Result')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    print(f"Final inertia: {kmeans.inertia_:.2f}")
    
    return kmeans

# Run the demo
if __name__ == "__main__":
    demo_kmeans()


# From https://chatgpt.com/g/g-iYSeH3EAI-website-generator/c/68128683-7660-8000-b357-5a6ef3d73fd7
class KmeansV2:
    def __init__(self, k: int):
        self.k = k

    def initialize_centroids(self, X, k):
        """Randomly initialize centroids from the data points"""
        indices = np.random.choice(X.shape[0], k, replace=False)
        return X[indices]

    def assign_clusters(self, X, centroids):
        """Assign each data point to the nearest centroid"""
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, labels, k):
        """Compute new centroids as the mean of assigned points"""
        return np.array([X[labels == i].mean(axis=0) for i in range(k)])

    def k_means(self, X, k, max_iters=100, tol=1e-4):
        centroids = self.initialize_centroids(X, k)
        for i in range(max_iters):
            old_centroids = centroids
            labels = self.assign_clusters(X, centroids)
            centroids = self.update_centroids(X, labels, k)
            # Check for convergence
            if np.linalg.norm(centroids - old_centroids) < tol:
                break
        return centroids, labels

# Example usage:
if __name__ == "__main__":
    # Create sample data
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

    # Run K-means
    k = 4
    centroids, labels = KmeansV1().k_means(X, k)

    # # Plot results
    # plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X')
    # plt.title("K-means Clustering")
    # plt.show()

