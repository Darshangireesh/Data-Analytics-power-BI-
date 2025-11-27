import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

def visualize_clusters_pca(data, clusters, output_dir='eda_plots'):
    """
    Visualizes clusters using PCA (2 Components).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=clusters, palette='viridis', s=100)
    plt.title('Clusters Visualized with PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(f'{output_dir}/pca_clusters.png')
    plt.close()

def visualize_cluster_profiles(original_data, clusters, output_dir='eda_plots'):
    """
    Visualizes the distribution of features for each cluster.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    data_with_clusters = original_data.copy()
    data_with_clusters['Cluster'] = clusters
    
    # Pairplot colored by cluster
    sns.pairplot(data_with_clusters, hue='Cluster', palette='viridis')
    plt.savefig(f'{output_dir}/cluster_pairplot.png')
    plt.close()
    
    # Boxplots for each feature
    features = [col for col in data_with_clusters.columns if col != 'Cluster']
    for col in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster', y=col, data=data_with_clusters, palette='viridis')
        plt.title(f'{col} by Cluster')
        plt.savefig(f'{output_dir}/boxplot_{col}.png')
        plt.close()

if __name__ == "__main__":
    # Test visualization
    try:
        scaled_df = pd.read_csv('processed_data/rfm_scaled.csv', index_col=0)
        original_df = pd.read_csv('processed_data/rfm_data.csv', index_col=0)
        
        # Mock clusters
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_df)
        
        visualize_clusters_pca(scaled_df, clusters)
        visualize_cluster_profiles(original_df, clusters)
        print("Visualizations generated.")
    except Exception as e:
        print(f"Error in visualization: {e}")
