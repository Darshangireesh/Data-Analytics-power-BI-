import os
import pandas as pd

def generate_report(profile, recommendations, output_dir='.'):
    """
    Generates a markdown report summarizing the findings.
    """
    report_content = "# Customer Segmentation Report\n\n"
    
    report_content += "## 1. Executive Summary\n"
    report_content += "This report presents the findings of the customer segmentation analysis using K-Means clustering. "
    report_content += "The goal was to identify distinct customer groups to enable targeted marketing strategies.\n\n"
    
    report_content += "## 2. Methodology\n"
    report_content += "- **Data Source**: Transactional data.\n"
    report_content += "- **Features**: Recency, Frequency, Monetary (RFM).\n"
    report_content += "- **Algorithm**: K-Means Clustering.\n"
    report_content += "- **Scaling**: Standard Scaler.\n\n"
    
    report_content += "## 3. Cluster Profiles\n"
    report_content += "The analysis identified the following customer segments:\n\n"
    
    for cluster_id, row in profile.iterrows():
        count = int(row['Count']) if 'Count' in row else 'N/A'
        report_content += f"### Cluster {cluster_id}\n"
        report_content += f"- **Size**: {count} customers\n"
        report_content += f"- **Average Recency**: {row['Recency']:.1f} days\n"
        report_content += f"- **Average Frequency**: {row['Frequency']:.1f} orders\n"
        report_content += f"- **Average Monetary Value**: ${row['Monetary']:.2f}\n"
        
        # Add recommendation if available
        if cluster_id in recommendations:
            rec = recommendations[cluster_id]
            report_content += f"- **Label**: {rec['Label']}\n"
            report_content += f"- **Strategy**: {', '.join(rec['Strategies'])}\n"
        
        report_content += "\n"
        
    report_content += "## 4. Visualizations\n"
    report_content += "### Cluster Distribution (PCA)\n"
    report_content += "![PCA Plot](eda_plots/pca_clusters.png)\n\n"
    
    report_content += "### Feature Distribution by Cluster\n"
    report_content += "![Pairplot](eda_plots/cluster_pairplot.png)\n\n"
    
    report_content += "## 5. Conclusion\n"
    report_content += "The segmentation provides actionable insights for marketing. "
    report_content += "We recommend implementing the suggested strategies to maximize customer value and retention.\n"
    
    with open(f'{output_dir}/Customer_Segmentation_Report.md', 'w') as f:
        f.write(report_content)
        
    return f'{output_dir}/Customer_Segmentation_Report.md'

if __name__ == "__main__":
    # Test report generation
    try:
        # Mock data
        data = {
            'Recency': [10, 100, 300],
            'Frequency': [10, 5, 1],
            'Monetary': [1000, 500, 100],
            'Count': [50, 30, 20]
        }
        profile = pd.DataFrame(data, index=[0, 1, 2])
        
        recs = {
            0: {'Label': 'Champions', 'Strategies': ['Reward']},
            1: {'Label': 'Potential', 'Strategies': ['Upsell']},
            2: {'Label': 'At Risk', 'Strategies': ['Re-engage']}
        }
        
        path = generate_report(profile, recs)
        print(f"Report generated at {path}")
    except Exception as e:
        print(f"Error in report generation: {e}")
