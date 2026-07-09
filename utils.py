import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import functools
import os
import re
import unicodedata
from pathlib import Path

# Get the absolute path to the project root directory
ROOT_DIR = Path(__file__).parent

# Cache for player images to avoid repeated requests
player_image_cache = {}

def clean_player_name(name):
    """
    Clean player names to fix encoding issues
    For example, "?lex Fern?ndez" -> "Álex Fernández"
    """
    if not isinstance(name, str):
        return str(name)
    
    # Fix common encoding issues
    name_map = {
        '?': '',  # Remove question marks from encoding errors
        'A?': 'Á', '?': 'á',
        'E?': 'É', '?': 'é',
        'I?': 'Í', '?': 'í',
        'O?': 'Ó', '?': 'ó',
        'U?': 'Ú', '?': 'ú',
        'N?': 'Ñ', '?': 'ñ'
    }
    
    for bad, good in name_map.items():
        name = name.replace(bad, good)
    
    # Normalize unicode characters
    try:
        name = unicodedata.normalize('NFC', name)
    except:
        pass
    
    # Remove any remaining non-printable or strange characters
    name = re.sub(r'[^\x20-\x7E\u00C0-\u017FáéíóúüÁÉÍÓÚÜñÑ]', '', name)
    
    return name.strip()

@functools.lru_cache(maxsize=1000)
def load_data(file_path=None, player_type=None):
    """
    Load player data from unified CSV file with robust encoding support
      Args:
        file_path: Optional path to a specific CSV file to load (deprecated)
        player_type: Optional filter for specific player positions (e.g., 'DF', 'MF', 'FW')
    """
    try:
        # Use unified dataset by default
        if file_path is None:
            csv_path = ROOT_DIR / 'all attributes.csv'
        else:
            csv_path = file_path if isinstance(file_path, Path) else Path(file_path)
            
        # Check if the file exists
        if not csv_path.exists():
            st.error(f"File not found: {csv_path}")
            return None
            
        # Try different encodings to handle special characters
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-8-sig']
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                
                # Clean player names to fix encoding issues
                if 'Player' in df.columns:
                    df['Player'] = df['Player'].apply(clean_player_name)
                
                # Filter by player type if specified
                if player_type:
                    # Filter players by primary position or multi-position that includes the type
                    df = df[df['Pos'].str.contains(player_type, na=False)]
                
                return df
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        # If none of the encodings work, try binary mode with replacement
        try:
            with open(csv_path, 'rb') as file:
                content = file.read()                # Replace invalid UTF-8 bytes with replacement character
                content_str = content.decode('utf-8', 'replace')
                
            # Convert to StringIO for pandas to read            
            import io
            string_io = io.StringIO(content_str)
            df = pd.read_csv(string_io)
            
            # Clean player names to fix encoding issues
            if 'Player' in df.columns:
                df['Player'] = df['Player'].apply(clean_player_name)
            
            # Filter by player type if specified
            if player_type:
                # Filter players by primary position or multi-position that includes the type
                df = df[df['Pos'].str.contains(player_type, na=False)]
            
            return df
        except Exception as binary_error:
            st.error(f"Error in binary mode: {binary_error}")
            return None
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@functools.lru_cache(maxsize=1000)
def load_player_images():
    """
    Load the player images data from CSV file
    """
    try:
        headshots_path = ROOT_DIR / 'player_headshots1.csv'
        return pd.read_csv(headshots_path)
    except Exception as e:
        st.error(f"Error loading player images data: {e}")
        return None

def get_player_image_url(player_name):
    """
    Get player image URL from the local CSV file
    """
    # Check cache first for previously found images
    if player_name in player_image_cache:
        return player_image_cache[player_name]
        
    try:
        # Get the headshots dataframe
        headshots_df = load_player_images()
        if headshots_df is None:
            return None
            
        # Look for the player in the headshots dataframe
        player_row = headshots_df[headshots_df['Player'] == player_name]
        
        if not player_row.empty and player_row['Image_URL'].iloc[0] != 'No image available':
            # Cache and return the image URL
            image_url = player_row['Image_URL'].iloc[0]
            player_image_cache[player_name] = image_url
            return image_url
            
        # If no image found, return None to use default silhouette
        return None
            
    except Exception as e:
        st.warning(f"Error getting image for {player_name}: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the dataframe for analysis
    """
    # Clean the player data
    df = df.dropna(subset=['Player'])
    
    # Convert all potential numeric columns to numeric, handling errors gracefully
    potential_numeric_cols = [
        'Age', 'Born', 'GCA', 'GCA90', 'Won', 'Lost', 'npxG+xAG', 'Att', 'Succ', 'SoT', 'FK', 'Sh', 'Gls', 'Ast',
        'PrgC', 'PrgP', 'PrgR', 'G+A', 'SCA', 'SCA90', '90s', 'TklW', 'Blocks', 'Int', 'Clr', 'Prgc', 'Tkl%', 'Crs'
    ]
    
    for col in potential_numeric_cols:
        if col in df.columns:
            # Convert to numeric, replacing non-numeric values with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Also handle any remaining numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def create_radar_chart(player_stats, max_stats, categories):
    """
    Create a radar chart for player statistics
    """
    # Number of variables
    N = len(categories)
    
    # Create angles for each stat
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    
    # Normalize player stats with safeguard against division by zero
    normalized_stats = []
    for stat, max_val in zip(player_stats, max_stats):
        # Handle edge case where max value is 0 to avoid division by zero
        if max_val == 0:
            normalized_stats.append(0)
        else:
            normalized_stats.append(stat / max_val)
    
    normalized_stats += normalized_stats[:1]  # Close the polygon
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
      # Draw the polygon
    ax.fill(angles, normalized_stats, 'skyblue', alpha=0.7)
    ax.plot(angles, normalized_stats, 'blue', linewidth=1)
      # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticklabels([])  # Hide the radial labels
    
    # Add title
    plt.title('Player Statistics Radar Chart', size=15, color='blue', y=1.1)
    
    return fig
