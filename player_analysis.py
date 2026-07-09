"""
Clean version of player_analysis.py to resolve import issues
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Global variables to cache preprocessing
_cached_data = None
_cached_scaler = None
_cached_features = None

def clear_cache():
    """Clear the cached preprocessing data"""
    global _cached_data, _cached_scaler, _cached_features
    _cached_data = None
    _cached_scaler = None
    _cached_features = None

def preprocess_data(df, features=None):
    """
    Preprocess data once and cache the results for reuse
    """
    global _cached_data, _cached_scaler, _cached_features
    
    if features is None:
        # Updated default features for new dataset - using columns that exist across positions
        features = ['Age', 'GCA', 'GCA90', 'Gls', 'Ast']
        
    # Make sure all features are present
    features = [f for f in features if f in df.columns]
    
    # Try to add more features if not enough exist
    if len(features) < 3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Rk', 'index', 'Born', 'Age']
        extra_features = [col for col in numeric_cols if col not in features and col not in exclude_cols]
        features.extend(extra_features)
    
    # Check if we have enough features
    if len(features) < 3:
        raise ValueError("Not enough features available for analysis")
    
    # Use median instead of 0 for missing values
    df_features = df[features].fillna(df[features].median())
    
    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_features)
    
    # Cache the results
    _cached_data = scaled_data
    _cached_scaler = scaler
    _cached_features = features
    
    return scaled_data, scaler, features

def find_similar_players(df, player_name, n_similar=5, features=None):
    """
    Find the top n similar players to the given player using cosine similarity
    """
    try:
        # Check if player exists in the dataset
        if player_name not in df['Player'].values:
            return None, "Player not found in the dataset"
        
        # Reset index to ensure sequential numbering from 0
        df_reset = df.reset_index(drop=True)
        
        # Get preprocessed data
        try:
            scaled_data, scaler, used_features = preprocess_data(df_reset, features)
        except ValueError as e:
            return None, f"Error preprocessing data: {str(e)}"
        
        # Get the player's index in the reset dataframe
        player_idx = df_reset[df_reset['Player'] == player_name].index[0]
        
        # Validate that the index is within bounds
        if player_idx >= len(scaled_data):
            return None, f"Player index {player_idx} is out of bounds for scaled data of size {len(scaled_data)}"
        
        # Calculate similarity
        target_player_scaled = scaled_data[player_idx].reshape(1, -1)
        similarities = cosine_similarity(target_player_scaled, scaled_data)[0]
        
        # Convert to percentages
        similarities_percent = (similarities * 100).round(2)
        
        # Get similarity scores for all players (excluding the player itself)
        sim_scores = list(enumerate(similarities_percent))
        
        # Sort the players by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_similar+1]
        
        # Get the player indices and similarity scores
        player_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Validate indices are within bounds
        max_idx = len(df_reset) - 1
        valid_indices = [idx for idx in player_indices if idx <= max_idx]
        valid_scores = [similarity_scores[i] for i, idx in enumerate(player_indices) if idx <= max_idx]
        
        if not valid_indices:
            return None, "No valid similar players found"
        
        # Create a dataframe with similar players using reset dataframe
        similar_players = df_reset.iloc[valid_indices].copy()
        
        # Add similarity scores
        similar_players['Similarity Score'] = [score / 100 for score in valid_scores]
        
        return similar_players, "Success"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error finding similar players: {str(e)}"

def find_optimal_clusters(df, max_clusters=10):
    """
    Find the optimal number of clusters using the Elbow Method
    """
    try:
        # Reset index to ensure sequential numbering from 0
        df_reset = df.reset_index(drop=True)
        
        # Use preprocessed data
        scaled_data, _, _ = preprocess_data(df_reset)
        
        # Calculate WCSS for different numbers of clusters
        wcss = []
        for n_clusters in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)
        
        # Create the Elbow Method plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='-')
        ax.set_title('Elbow Method for Optimal Number of Clusters')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('WCSS (Within-Cluster Sum of Squares)')
        ax.grid(True)
        
        # Find the elbow point
        diffs = [wcss[i-1] - wcss[i] for i in range(1, len(wcss))]
        acceleration = [diffs[i-1] - diffs[i] for i in range(1, len(diffs))]
        optimal_clusters = acceleration.index(max(acceleration)) + 2
        
        return optimal_clusters, fig
        
    except Exception as e:
        return None, f"Error in clustering analysis: {str(e)}"

def cluster_players(df, n_clusters=None):
    """
    Cluster players using K-means
    """
    try:
        # Reset index to ensure sequential numbering from 0
        df_reset = df.reset_index(drop=True)
        
        if n_clusters is None:
            result = find_optimal_clusters(df_reset)
            if isinstance(result[0], int):
                n_clusters = result[0]
            else:
                return None, "Error finding optimal clusters"
        
        # Use preprocessed data
        scaled_data, _, _ = preprocess_data(df_reset)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_clustered = df_reset.copy()
        df_clustered['Cluster'] = kmeans.fit_predict(scaled_data)
        
        return df_clustered, "Success"
        
    except Exception as e:
        return None, f"Error in clustering: {str(e)}"

# Print confirmation when module is imported successfully
print("player_analysis.py loaded successfully")
