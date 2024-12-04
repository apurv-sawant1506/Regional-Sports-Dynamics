# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch

# Data Preprocessing: Merge all datasets
def preprocess_data(population_data, sports_data, sentiment_data, economic_data, demographic_data, historical_data):
    merged_data = pd.merge(population_data, sports_data, on='Region')
    merged_data = pd.merge(merged_data, sentiment_data, on='Region')
    merged_data = pd.merge(merged_data, economic_data, on='Region')
    merged_data = pd.merge(merged_data, demographic_data, on='Region')
    historical_summary = historical_data.groupby('Region').agg({
        'Wins': 'mean',
        'Losses': 'mean',
        'Playoffs': 'sum'
    }).reset_index()
    merged_data = pd.merge(merged_data, historical_summary, on='Region', suffixes=('', '_historical'))
    merged_data.fillna(0, inplace=True)
    return merged_data

# Feature Engineering: Calculate WLDI, PIC, and Standardization
def calculate_metrics(data):
    data['WLDI'] = data['Wins'] / (data['Wins'] + data['Losses'] + 1e-5)
    data['PIC'] = data['Population'] / (data['Wins'] + 1e-5)
    data['Standardized_WL'] = (data['Wins'] - data['Losses']) / (data['Wins'] + data['Losses'] + 1e-5)
    return data

# Hierarchical Clustering
def cluster_regions(data, features):
    clustering_model = AgglomerativeClustering(n_clusters=5)
    cluster_labels = clustering_model.fit_predict(data[features])
    data['Cluster'] = cluster_labels
    return data

# Dimensionality Reduction
def reduce_dimensions(data, features):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data[features])
    tsne = TSNE(n_components=2)
    tsne_data = tsne.fit_transform(reduced_data)
    return tsne_data

# Train Prediction Models with Bootstrapping
def train_models(data, target):
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Bayesian Regression
    bayesian_model = BayesianRidge()
    bayesian_model.fit(X_train, y_train)
    y_pred = bayesian_model.predict(X_test)
    print("Bayesian Model R2 Score:", r2_score(y_test, y_pred))
    
    # Random Forest with Bootstrapping
    rf_model = RandomForestRegressor(oob_score=True, bootstrap=True)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))
    
    return bayesian_model, rf_model

# Graph Neural Network Analysis
class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels=5, out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def train_gnn(graph_data):
    edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
    x = torch.tensor(graph_data['node_features'], dtype=torch.float)
    y = torch.tensor(graph_data['targets'], dtype=torch.float).unsqueeze(1)
    
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    model = GNNModel()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Visualization Functions
def visualize_clusters(data, x_col, y_col):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, hue='Cluster', data=data, palette='viridis')
    plt.title("Cluster Visualization")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title="Cluster")
    plt.show()

def plot_league_comparisons(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='League', y='Standardized_WL', data=data)
    plt.title("League Comparison: Standardized Win/Loss Ratios")
    plt.xlabel("League")
    plt.ylabel("Standardized Win/Loss Ratio")
    plt.show()

def sentiment_correlation(data):
    correlation = data[['Positive_Sentiment', 'Negative_Sentiment', 'Wins']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title("Sentiment vs Performance Correlation")
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Load datasets
    population_data = pd.read_csv("population_data.csv")
    sports_data = pd.read_csv("sports_data.csv")
    sentiment_data = pd.read_csv("sentiment_data.csv")
    economic_data = pd.read_csv("economic_data.csv")
    demographic_data = pd.read_csv("demographic_data.csv")
    historical_data = pd.read_csv("historical_performance_data.csv")
    graph_data = pd.read_csv("graph_data.csv")
    
    # Preprocess and calculate metrics
    data = preprocess_data(population_data, sports_data, sentiment_data, economic_data, demographic_data, historical_data)
    data = calculate_metrics(data)
    
    # Cluster regions
    data = cluster_regions(data, features=['Population', 'GDP', 'Median_Income', 'Wins', 'Losses'])
    
    # Dimensionality reduction for visualization
    tsne_data = reduce_dimensions(data, features=['Population', 'Wins', 'Losses', 'GDP', 'Median_Income'])
    
    # Train prediction models
    train_models(data, target='Wins')
    
    # Visualizations
    visualize_clusters(data, x_col='Population', y_col='GDP')
    plot_league_comparisons(data)
    sentiment_correlation(data)
    
    # Prepare graph data for GNN
    graph_data_dict = {
        'node_features': data[['Population', 'GDP', 'Median_Income', 'Wins', 'Losses']].values,
        'edge_index': graph_data[['Source', 'Target']].values,
        'targets': data['Wins'].values
    }
    train_gnn(graph_data_dict)