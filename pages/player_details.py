import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

# Get the absolute path to the project root directory
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Now import the utilities
from utils import load_data, preprocess_data, get_player_image_url, create_radar_chart

# Import functions from player_analysis with error handling
try:
    from player_analysis import find_similar_players, find_optimal_clusters, cluster_players, clear_cache
    print("Successfully imported functions from player_analysis")
except ImportError as e:
    import streamlit as st  # Make sure st is imported for the error message
    st.error(f"Error importing from player_analysis modules: {e}")
    # Define fallback functions to avoid breaking the app
    def find_similar_players(df, player_name, n_similar=5, features=None):
        st.error("Player analysis module could not be loaded")
        return None, "Module import error"
    
    def find_optimal_clusters(df, max_clusters=10):
        st.error("Player analysis module could not be loaded")
        return None, "Module import error"
    
    def cluster_players(df, n_clusters=None):
        st.error("Player analysis module could not be loaded")
        return None, "Module import error"
        
    def clear_cache():
        pass

def show_player_details_page():
    # Check if a player has been selected
    if 'selected_player' not in st.session_state:
        st.error("No player selected. Please go back to the home page.")
        if st.button("Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        return
    
    player_name = st.session_state.selected_player
    
    # Clear the player analysis cache to ensure we use the new data
    from player_analysis import clear_cache
    clear_cache()
    # Also clear the data loading cache
    load_data.cache_clear()
    
    # --- UPDATED: Use unified dataset ---    
    # Define player types with their position filters and key stats    
    player_types = {
        'Defenders (DF)': {
            'filter': 'DF',
            'key_stats': ['90s', 'TklW', 'Blocks', 'Int', 'Clr', 'Prgc', 'G+A', 'Tkl%', 'Crs', 'Att', 'Succ', 'Won']
        },
        'Midfielders (MF)': {
            'filter': 'MF',
            'key_stats': ['PrgC', 'PrgP', 'PrgR', 'G+A', 'SCA', 'SCA90', '90s']
        },
        'Forwards (FW)': {
            'filter': 'FW',
            'key_stats': ['GCA', 'GCA90', 'Won', 'Lost', 'npxG+xAG', 'Att', 'Succ', 'SoT', 'FK', 'Sh', 'Gls', 'Ast']
        },
        'Goalkeepers (GK)': {
            'filter': 'GK',
            'key_stats': ['GCA', 'GCA90', 'Gls', 'Ast', 'Age']
        }
    }
    
    # First load the full dataset to determine player position
    try:
        full_df = load_data()
        if full_df is None:
            st.error("Failed to load player data")
            return
            
        # Get the player data to detect their position
        initial_player_data = full_df[full_df['Player'] == player_name]
        
        if initial_player_data.empty:
            st.error(f"Player {player_name} not found in the dataset")
            return
            
        # Automatically determine player type based on position
        player_position = initial_player_data['Pos'].iloc[0] if 'Pos' in initial_player_data.columns else ""
        
        # Default to 'Forwards (FW)' if position cannot be determined
        dataset_type = 'Forwards (FW)'
        
        if player_position:
            # Extract first two characters (e.g., DF, MF, FW, GK)
            main_position = player_position[:2] if len(player_position) >= 2 else player_position
            
            # Map position to dataset_type
            if 'DF' in main_position and 'Defenders (DF)' in player_types:
                dataset_type = 'Defenders (DF)'
            elif 'MF' in main_position and 'Midfielders (MF)' in player_types:
                dataset_type = 'Midfielders (MF)' 
            elif 'FW' in main_position and 'Forwards (FW)' in player_types:
                dataset_type = 'Forwards (FW)'
            elif 'GK' in main_position and 'Goalkeepers (GK)' in player_types:
                dataset_type = 'Goalkeepers (GK)'
        
    except Exception as e:
        st.error(f"Error determining player type: {e}")
        # Default to Forwards if there's an error
        dataset_type = 'Forwards (FW)'
    
    # Get position filter and stats based on selected type
    position_filter = player_types[dataset_type]['filter']
    key_stats = player_types[dataset_type]['key_stats']
    similarity_features = key_stats.copy()
    
    # Load and preprocess data for the selected type using unified dataset
    try:
        df = load_data(player_type=position_filter)
        
        if df is None:
            st.error(f"Failed to load data for {dataset_type}")
            return
            
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return
    
    df = preprocess_data(df)

    # Get player data
    player_data = df[df['Player'] == player_name]
    
    if player_data.empty:
        st.error(f"Player {player_name} not found in the dataset.")
        return
    
    # Set up the layout with columns
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Display player image
        try:
            image_url = get_player_image_url(player_name)
            if image_url and isinstance(image_url, str):
                try:
                    # Display the actual player image with a styled container
                    st.markdown(
                        f"""
                        <div style="
                            width: 150px;
                            height: 150px;
                            border-radius: 10px;
                            overflow: hidden;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                            background: linear-gradient(145deg, #1a1a1a, #2a2a2a);
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        ">
                            <img src="{image_url}" style="width: 100%; height: 100%; object-fit: cover;">
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                except Exception:
                    # Display default person icon with gradient background
                    st.markdown(
                        """
                        <div style="
                            width: 150px;
                            height: 150px;
                            border-radius: 10px;
                            overflow: hidden;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                            background: linear-gradient(145deg, #1e3c72, #2a5298);
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            color: rgba(255,255,255,0.8);
                        ">
                            <svg width="60" height="60" viewBox="0 0 448 512" fill="currentColor">
                                <path d="M224 256A128 128 0 1 0 224 0a128 128 0 1 0 0 256zm-45.7 48C79.8 304 0 383.8 0 482.3C0 498.7 13.3 512 29.7 512H418.3c16.4 0 29.7-13.3 29.7-29.7C448 383.8 368.2 304 269.7 304H178.3z"/>
                            </svg>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                # Display default person icon with gradient background
                st.markdown(
                    """
                    <div style="
                        width: 150px;
                        height: 150px;
                        border-radius: 10px;
                        overflow: hidden;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        background: linear-gradient(145deg, #1e3c72, #2a5298);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: rgba(255,255,255,0.8);
                    ">
                        <svg width="60" height="60" viewBox="0 0 448 512" fill="currentColor">
                            <path d="M224 256A128 128 0 1 0 224 0a128 128 0 1 0 0 256zm-45.7 48C79.8 304 0 383.8 0 482.3C0 498.7 13.3 512 29.7 512H418.3c16.4 0 29.7-13.3 29.7-29.7C448 383.8 368.2 304 269.7 304H178.3z"/>
                        </svg>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception:
            # Display default person icon with gradient background
            st.markdown(
                """
                <div style="
                    width: 150px;
                    height: 150px;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    background: linear-gradient(145deg, #1e3c72, #2a5298);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: rgba(255,255,255,0.8);
                ">
                    <svg width="60" height="60" viewBox="0 0 448 512" fill="currentColor">
                        <path d="M224 256A128 128 0 1 0 224 0a128 128 0 1 0 0 256zm-45.7 48C79.8 304 0 383.8 0 482.3C0 498.7 13.3 512 29.7 512H418.3c16.4 0 29.7-13.3 29.7-29.7C448 383.8 368.2 304 269.7 304H178.3z"/>
                    </svg>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Display player info
        st.markdown(f"## {player_name}")
        
        if 'Nation' in player_data.columns:
            st.write(f"**Nationality:** {player_data['Nation'].values[0]}")
        
        if 'Pos' in player_data.columns:
            st.write(f"**Position:** {player_data['Pos'].values[0]}")
        
        if 'Squad' in player_data.columns:
            st.write(f"**Team:** {player_data['Squad'].values[0]}")
        
        if 'Comp' in player_data.columns:
            st.write(f"**League:** {player_data['Comp'].values[0]}")
        
        if 'Born' in player_data.columns:
            born_value = player_data['Born'].values[0]
            # Convert to integer if it's a float (to remove decimal point)
            if isinstance(born_value, (float, int)):
                born_value = int(born_value)
            st.write(f"**Born:** {born_value}")
    
    with col2:
        st.markdown("## Player Statistics")
        
        # Create tabs for different statistics views
        tabs = st.tabs(["Key Stats", "Player Comparison", "Clustering"])
        
        with tabs[0]:
            # Display key stats in a table
            # Updated stats list based on available columns in DF players.csv
            stats_to_display = [col for col in key_stats if col in player_data.columns]
            if stats_to_display:
                stats_df = player_data[stats_to_display].T.reset_index()
                stats_df.columns = ['Stat', 'Value']
                
                # Format numeric values to have exactly 2 decimal places
                for i, value in enumerate(stats_df['Value']):
                    if isinstance(value, (float, int)):
                        # Special handling for 'Born' field to show as integer
                        if stats_df.loc[i, 'Stat'] == 'Born':
                            stats_df.loc[i, 'Value'] = f"{int(value)}"
                        else:
                            stats_df.loc[i, 'Value'] = f"{value:.2f}"
                
                st.table(stats_df)
            else:
                st.warning("No statistics available for this player.")
        
        # --- Player Comparison and Radar Chart ---
        with tabs[1]:
            col_chart, col_similar = st.columns([3, 2])
            
            with col_chart:
                st.markdown("### Player Radar Chart")
                
                if 'comparison_player' not in st.session_state:
                    st.session_state.comparison_player = None
                
                if all(stat in player_data.columns for stat in key_stats):
                    # Safely extract stats, ensuring numeric values
                    stats = []
                    for stat in key_stats:
                        try:
                            val = pd.to_numeric(player_data[stat].iloc[0], errors='coerce')
                            stats.append(0 if pd.isna(val) else val)
                        except Exception:
                            stats.append(0)
                    
                    # Safely calculate max stats, ensuring numeric values only
                    max_stats = []
                    for stat in key_stats:
                        try:
                            # Convert to numeric and get max, handling any remaining string values
                            numeric_series = pd.to_numeric(df[stat], errors='coerce').fillna(0)
                            max_val = numeric_series.max()
                            max_stats.append(max_val)
                        except Exception:
                            # Fallback to 1 if we can't get a proper max value
                            max_stats.append(1)
                    
                    if st.session_state.comparison_player is not None:
                        # First try to find comparison player in current filtered dataset
                        comp_player_data = df[df['Player'] == st.session_state.comparison_player]
                        
                        # If not found in filtered data, load from full dataset
                        if comp_player_data.empty:
                            try:
                                full_df = load_data()  # Load full dataset without position filter
                                comp_player_data = full_df[full_df['Player'] == st.session_state.comparison_player]
                            except Exception as e:
                                st.error(f"Error loading full dataset for comparison: {e}")
                                comp_player_data = pd.DataFrame()
                        
                        # Check if comparison player has different position and adjust key_stats accordingly
                        if not comp_player_data.empty:
                            comp_position = comp_player_data['Pos'].iloc[0] if 'Pos' in comp_player_data.columns else ''
                            
                            # Determine appropriate stats for comparison player's position
                            comp_key_stats = key_stats  # Default to current player's stats
                            if 'DF' in comp_position:
                                comp_key_stats = ['90s', 'TklW', 'Blocks', 'Int', 'Clr', 'Prgc', 'G+A', 'Tkl%', 'Crs', 'Att', 'Succ', 'Won']
                            elif 'MF' in comp_position:
                                comp_key_stats = ['PrgC', 'PrgP', 'PrgR', 'G+A', 'SCA', 'SCA90', '90s']
                            elif 'FW' in comp_position:
                                comp_key_stats = ['GCA', 'GCA90', 'Won', 'Lost', 'npxG+xAG', 'Att', 'Succ', 'SoT', 'FK', 'Sh', 'Gls', 'Ast']
                            
                            # Use common stats between both players for comparison
                            common_stats = [stat for stat in key_stats if stat in comp_key_stats and stat in comp_player_data.columns]
                            
                            # If no common stats, use any available stats from current player's key_stats
                            if not common_stats:
                                common_stats = [stat for stat in key_stats if stat in comp_player_data.columns]
                            
                            if common_stats and all(stat in comp_player_data.columns for stat in common_stats):
                                # Use common stats for both players - ensure numeric values with proper formatting
                                stats_filtered = []
                                for stat in common_stats:
                                    try:
                                        val = pd.to_numeric(player_data[stat].iloc[0], errors='coerce')
                                        # Handle 'Born' field specially
                                        if stat == 'Born' and not pd.isna(val):
                                            val = int(val)
                                        stats_filtered.append(0 if pd.isna(val) else val)
                                    except Exception:
                                        stats_filtered.append(0)
                                
                                comp_stats = []
                                for stat in common_stats:
                                    try:
                                        val = pd.to_numeric(comp_player_data[stat].iloc[0], errors='coerce')
                                        # Handle 'Born' field specially
                                        if stat == 'Born' and not pd.isna(val):
                                            val = int(val)
                                        comp_stats.append(0 if pd.isna(val) else val)
                                    except Exception:
                                        comp_stats.append(0)
                                
                                max_stats_filtered = []
                                for stat in common_stats:
                                    try:
                                        numeric_series = pd.to_numeric(df[stat], errors='coerce').fillna(0)
                                        max_val = numeric_series.max()
                                        max_stats_filtered.append(max_val if max_val > 0 else 1)
                                    except Exception:
                                        max_stats_filtered.append(1)
                                
                                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                                N = len(common_stats)
                                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                                angles += angles[:1]
                                ax.set_xticks(angles[:-1])
                                ax.set_xticklabels(common_stats)
                                ax.set_yticklabels([])
                                normalized_stats = [stat / max_val if max_val else 0 for stat, max_val in zip(stats_filtered, max_stats_filtered)]
                                normalized_stats += normalized_stats[:1]
                                ax.fill(angles, normalized_stats, 'skyblue', alpha=0.25)
                                ax.plot(angles, normalized_stats, 'blue', linewidth=1, label=player_name)
                                comp_normalized_stats = [stat / max_val if max_val else 0 for stat, max_val in zip(comp_stats, max_stats_filtered)]
                                comp_normalized_stats += comp_normalized_stats[:1]
                                ax.fill(angles, comp_normalized_stats, 'lightgreen', alpha=0.25)
                                ax.plot(angles, comp_normalized_stats, 'green', linewidth=1, label=st.session_state.comparison_player)
                                ax.legend(loc='upper right')
                                st.pyplot(fig)
                                if st.button("Clear Comparison"):
                                    st.session_state.comparison_player = None
                                    st.rerun()
                            else:
                                st.warning("Cannot find common statistics for comparison between these players.")
                        else:
                            st.warning("Cannot load comparison data for the selected player.")
                    else:
                        fig = create_radar_chart(stats, max_stats, key_stats)
                        st.pyplot(fig)
                else:
                    st.warning("Not enough statistics available for radar chart.")
                    
            with col_similar:
                st.markdown("### Similar Players (Same Position)")
                
                # --- Similar Players (pass features) ---
                from player_analysis import find_similar_players as find_similar_players_func
                
                # Get the player's position
                player_position = player_data['Pos'].iloc[0] if 'Pos' in player_data.columns else ""
                
                # Filter the dataframe to only include players in the same position
                if player_position:
                    # Get the main position (first two characters, e.g., DF, MF, FW)
                    main_position = player_position[:2] if len(player_position) >= 2 else player_position
                    position_filtered_df = df[df['Pos'].str.contains(main_position, na=False)]
                else:
                    position_filtered_df = df
                
                # Find similar players within the same position group
                similar_players, message = find_similar_players_func(position_filtered_df, player_name, n_similar=5, features=similarity_features)
                if similar_players is not None:
                    for idx, row in similar_players.iterrows():
                        # Format similarity score with consistent 2 decimal places
                        similarity = row['Similarity Score'] * 100
                        similarity_formatted = f"{similarity:.2f}%"
                        player_img = get_player_image_url(row['Player'])
                        position = row['Pos'] if 'Pos' in row else ""
                        html = f"""
                        <div style=\"display: flex; align-items: center; margin-bottom: 10px; padding: 5px; background-color: #1E1E1E; border-radius: 5px;\">
                            <div style=\"flex: 1; text-align: center;\">
                                <img src=\"{player_img}\" style=\"width: 50px; height: 50px; border-radius: 50%;\">
                            </div>
                            <div style=\"flex: 2; padding-left: 10px;\">
                                <p style=\"margin: 0; font-weight: bold;\">{row['Player']}</p>
                                <p style=\"margin: 0; color: #AAA; font-size: 12px;\">{position}</p>
                            </div>
                            <div style=\"flex: 1; display: flex; flex-direction: column; align-items: center;\">
                                <div style=\"width: 40px; height: 40px; border-radius: 50%; background: conic-gradient(#00E676 {similarity}%, #2E3033 0%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px; margin-bottom: 5px;\">
                                    {similarity_formatted}
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(html, unsafe_allow_html=True)
                        if st.button("Compare", key=f"compare_{idx}"):
                            st.session_state.comparison_player = row['Player']
                            st.rerun()
                        st.markdown("<hr>", unsafe_allow_html=True)
                else:
                    st.warning(message)
                    
        with tabs[2]:
            # Display clustering analysis
            st.markdown("### Player Clustering Analysis")
            
            # Find optimal number of clusters
            result = find_optimal_clusters(df)
            
            if isinstance(result, tuple) and len(result) == 2:
                n_clusters, fig = result
                
                if isinstance(n_clusters, int):
                    st.markdown(f"#### Optimal number of clusters: {n_clusters}")
                    st.pyplot(fig)
                else:
                    st.warning(f"Error finding optimal clusters: {n_clusters}")
                    n_clusters = 3  # Default number of clusters as fallback
            else:
                st.warning("Error determining optimal number of clusters")
                n_clusters = 3  # Default number of clusters as fallback
            
            # Apply clustering            
            df_clustered, cluster_message = cluster_players(df, n_clusters)
            
            if df_clustered is not None:
                try:
                    # Get current player's cluster
                    player_cluster = df_clustered[df_clustered['Player'] == player_name]['Cluster'].values[0]
                    
                    # Get other players in the same cluster
                    cluster_members = df_clustered[df_clustered['Cluster'] == player_cluster]
                    st.markdown(f"#### Players in the same cluster (Cluster {player_cluster}):")
                    
                    # Display cluster members (excluding the current player)
                    cluster_members = cluster_members[cluster_members['Player'] != player_name]
                    
                    display_cols = ['Player', 'Nation', 'Pos', 'Squad']
                    display_cols = [col for col in display_cols if col in cluster_members.columns]
                    
                    if not cluster_members.empty:
                        st.table(cluster_members[display_cols].head(5))  # Show only top 5 to avoid clutter
                    else:
                        st.warning("No other players found in the same cluster.")
                except Exception as e:
                    st.error(f"Error displaying cluster information: {str(e)}")
            else:
                st.warning(cluster_message)
    
    # Button to go back to home
    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()

if __name__ == "__main__":
    show_player_details_page()
