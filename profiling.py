import pandas as pd
import os

def profile_clusters(data, clusters):
    """
    Profiles clusters by calculating mean values of features.
    """
    data['Cluster'] = clusters
    profile = data.groupby('Cluster').mean()
    profile['Count'] = data['Cluster'].value_counts()
    return profile

def generate_recommendations(profile):
    """
    Generates marketing recommendations based on cluster profiles.
    Assumes 3 clusters and specific characteristics (High Value, Frequent, etc.)
    This is a rule-based generation and might need adjustment based on actual cluster centers.
    """
    recommendations = {}
    
    # Sort clusters by Monetary value to identify tiers
    sorted_clusters = profile.sort_values(by='Monetary', ascending=False)
    
    for cluster_id, row in sorted_clusters.iterrows():
        rec = []
        if row['Monetary'] > profile['Monetary'].mean() and row['Frequency'] > profile['Frequency'].mean():
            label = "Champions"
            rec.append("Reward with loyalty programs.")
            rec.append("Offer exclusive early access to new products.")
        elif row['Recency'] > profile['Recency'].mean():
            label = "At Risk"
            rec.append("Send re-engagement emails with discounts.")
            rec.append("Ask for feedback.")
        elif row['Monetary'] < profile['Monetary'].mean() and row['Frequency'] < profile['Frequency'].mean():
            label = "Low Value"
            rec.append("Nurture with educational content.")
            rec.append("Offer bundle deals to increase basket size.")
        else:
            label = "Potential Loyalists"
            rec.append("Upsell higher value products.")
            rec.append("Offer membership benefits.")
            
        recommendations[cluster_id] = {
            'Label': label,
            'Characteristics': f"R: {row['Recency']:.1f}, F: {row['Frequency']:.1f}, M: ${row['Monetary']:.2f}",
            'Strategies': rec
        }
        
    return recommendations

if __name__ == "__main__":
    # Test profiling
    try:
        df = pd.read_csv('processed_data/rfm_with_clusters.csv', index_col=0)
        clusters = df['Cluster']
        # Drop cluster col from data for profiling calculation if needed, but groupby handles it.
        # We need original RFM values, not scaled ones.
        # Assuming rfm_with_clusters.csv has original values + cluster label.
        # If it was saved from scaled, we need to load original and add clusters.
        
        # Let's check if rfm_with_clusters has original values. 
        # In main.py: rfm_df['Cluster'] = clusters -> rfm_df was original RFM. Correct.
        
        profile = profile_clusters(df, clusters)
        print("Cluster Profile:")
        print(profile)
        
        recs = generate_recommendations(profile)
        print("\nRecommendations:")
        for cid, info in recs.items():
            print(f"Cluster {cid} ({info['Label']}): {info['Strategies']}")
            
    except Exception as e:
        print(f"Error in profiling: {e}")
