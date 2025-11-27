import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

def determine_optimal_clusters(data, max_k=10, output_dir='eda_plots'):
    """
    Determines the optimal number of clusters using Elbow Method and Silhouette Score.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    inertia = []
    silhouette_scores = []
    K = range(2, max_k + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
        
    # Plot Elbow Method
    plt.figure(figsize=(10, 5))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.savefig(f'{output_dir}/elbow_method.png')
    plt.close()
    
    # Plot Silhouette Scores
    plt.figure(figsize=(10, 5))
    plt.plot(K, silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score For Optimal k')
    plt.savefig(f'{output_dir}/silhouette_score.png')
    plt.close()
    
    return inertia, silhouette_scores

def perform_clustering(data, n_clusters):
    """
    Performs K-Means clustering on the data.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

if __name__ == "__main__":
    # Test clustering
    try:
        df = pd.read_csv('processed_data/rfm_scaled.csv', index_col=0)
        determine_optimal_clusters(df)
        clusters, _ = perform_clustering(df, n_clusters=3)
        print(f"Clustering completed. Assigned {len(clusters)} labels.")
    except Exception as e:
        print(f"Error in clustering: {e}")
