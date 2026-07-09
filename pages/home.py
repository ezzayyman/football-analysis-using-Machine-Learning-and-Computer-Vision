import streamlit as st
import sys
import os
import base64
from pathlib import Path
from PIL import Image
import io

# Get the absolute path to the project root directory
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Now import the utilities
from utils import load_data, preprocess_data

def get_base64_encoded_image(image_path):
    """Get the base64 encoded image to use in CSS background"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        # Fallback to a default image if there's an error
        print(f"Error loading background image: {e}")
        return None

def show_home_page():
    # Set up a container to center all content
    with st.container():
        # Get the background image as base64
        bg_image_path = ROOT_DIR / 'Home page.jpg'
        background_image = get_base64_encoded_image(bg_image_path)        # Create a background image using the local image
        css = f"""
            <style>
            /* Hide the top bar */
            header {{
                visibility: hidden !important;
                display: none !important;
            }}
            
            /* Hide the hamburger menu */
            section[data-testid="stSidebarNav"] {{
                display: none !important;
            }}

            /* Hide other controls in the top area */
            button[kind="header"] {{
                display: none !important;
            }}
            
            .stApp {{
                background-image: url('data:image/jpeg;base64,{background_image}');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            
            /* Center the content */
            /* Main page container styling */
            #root > div:first-child {{
                padding-top: 0 !important;
            }}
        
            /* Hide fullscreen button */
            button[title="View fullscreen"] {{
                display: none !important;
            }}
            
            .main-content {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                max-width: 900px;
                margin: 0 auto;
                padding: 4rem 2rem;
                background-color: rgba(13, 17, 23, 0.85);
                border-radius: 10px;
                margin-top: 5vh;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                color: white;
                text-align: center;
            }}
            
            /* Ensure all text elements are centered */
            .main-content * {{
                text-align: center;
                margin-left: auto;
                margin-right: auto;
            }}
            
            h1 {{
                font-size: 3.5rem !important;
                font-weight: 700 !important;
                margin-bottom: 1.5rem !important;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                color: white !important;
            }}            .subtitle {{
                font-size: 1.3rem !important;
                margin-bottom: 2rem !important;
                line-height: 1.6 !important;
                color: #d1d1d1 !important;
                max-width: 700px;
                margin-left: auto;
                margin-right: auto;
                display: block;
                width: 100%;
                text-align: center !important;
            }}
            
            /* Center the selectbox and labels */
            .stSelectbox, .stSelectbox > div, label {{
                text-align: center !important;
                width: 100%;
                display: flex;
                justify-content: center;
            }}
            
            /* Style buttons */
            .stButton > button {{
                background-color: #4CAF50 !important;
                color: white !important;
                font-weight: bold !important;
                border: none !important;
                padding: 0.5rem 1.5rem !important;
                font-size: 1.1rem !important;
                transition: all 0.3s !important;
                width: 100%;
                max-width: 300px;
            }}
            
            .stButton > button:hover {{
                background-color: #45a049 !important;
                box-shadow: 0 0 10px rgba(76, 175, 80, 0.5) !important;
                transform: translateY(-2px);
            }}
            </style>
            
            <div class="main-content">
            """
        st.markdown(css, unsafe_allow_html=True)        # Create a full-width container for title and subtitle
        st.markdown('<div style="width:100%; text-align:center; margin:0 auto; padding-top:2rem;">', unsafe_allow_html=True)
          
        # Title and subtitle
        st.markdown("<h1 style='text-align:center;width:100%;margin-bottom:2rem;'>Football Player Scout</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle' style='text-align:center;width:80%;margin:0 auto 3rem auto;'>Explore comprehensive player comparisons, uncover hidden football stars, and easily analyze playing styles.</p>", unsafe_allow_html=True)
        
        # Close the container
        st.markdown('</div>', unsafe_allow_html=True)
          # Initialize dataset type and position filter with default values
        dataset_type = 'All Players'
        position_filter = None
        
        # Load the unified dataset with optional position filtering
        df = load_data(player_type=position_filter)
        if df is None:
            st.error("Failed to load player data. Please check if the CSV file exists.")
            return
        
        df = preprocess_data(df)
        
        # Store the selected player type and data in the session state
        st.session_state.dataset_type = dataset_type
        st.session_state.position_filter = position_filter        # Add a search box for players with custom styling
        st.markdown("<h3 style='margin-top: 2rem; margin-bottom: 1.5rem; text-align:center; width:100%;'>Search & Select a Player:</h3>", unsafe_allow_html=True)
        
        # Create columns to center the search field
        col1, col2, col3 = st.columns([1, 3, 1])
        
        # Get all player names for the dropdown
        player_names = sorted(df['Player'].unique().tolist())
        
    # Create a searchable dropdown in the center column
    with col2:
        selected_player = st.selectbox(
            "Select a player...", 
            player_names,
            key=f"player_select_{dataset_type}",
            label_visibility="collapsed"
        )
    
    # Add a container to better center the button
    st.markdown('<div style="display: flex; justify-content: center; width: 100%; margin: 2rem auto;">', unsafe_allow_html=True)
    
    # Button to view player details - using a narrower column setup for better centering
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        if st.button("Analyze Player", use_container_width=True):
            if selected_player:
                # Store the selected player and dataset type in session state
                st.session_state.selected_player = selected_player
                st.session_state.dataset_type = dataset_type
                st.session_state.page = "player_details"
                st.rerun()
            else:
                st.error("Please select a player")
                
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Close the main content div
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    show_home_page()
