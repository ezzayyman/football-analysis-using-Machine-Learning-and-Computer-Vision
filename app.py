import streamlit as st
import os
import sys
import traceback
from pathlib import Path

# Get the absolute path to the project root directory
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

# Print debugging information
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"ROOT_DIR: {ROOT_DIR}")

# Make sure modules are importable
success = True

# Try importing player_analysis
try:
    import player_analysis
    print("Successfully imported player_analysis")
except Exception as e:
    print(f"Error importing player_analysis: {e}")
    traceback.print_exc()
    success = False

# Import the pages
try:
    from pages.home import show_home_page
    print("Successfully imported home page")
except Exception as e:
    print(f"Error importing home page: {e}")
    traceback.print_exc()
    success = False

try:
    from pages.player_details import show_player_details_page
    print("Successfully imported player details page")
except Exception as e:
    print(f"Error importing player details page: {e}")
    traceback.print_exc()
    success = False

# Import the new video analysis page
try:
    from pages.video_analysis import show_video_analysis_page
    print("Successfully imported video analysis page")
except ImportError as e:
    print(f"Error importing video analysis page: {e}")
    traceback.print_exc()
    success = False
except Exception as e:
    print(f"An unexpected error occurred during video analysis page import: {e}")
    traceback.print_exc()
    success = False


# Set page configuration
st.set_page_config(
    page_title="Football Analysis App", # Updated title
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded" # Expanded by default to show navigation
)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Create a sidebar with navigation options
with st.sidebar:
    st.title("Navigation")
    if st.button("Home"):
        st.session_state.page = "home"
        # No need for st.rerun() here, Streamlit handles it

    if st.button("Player Details"):
        st.session_state.page = "player_details"

    # Add button for the new Video Analysis page
    if st.button("Video Analysis"):
        st.session_state.page = "video_analysis"

    st.markdown("---")
    st.markdown("## About")
    st.markdown("""
    This application provides tools for football analysis:
    - Player scouting and similarity analysis.
    - Automated video analysis for player tracking, team assignment, and ball control.
    """)

# Render the appropriate page based on the session state
if st.session_state.page == "home":
    if 'show_home_page' in globals():
        show_home_page()
    else:
        st.error("Home page failed to load.")
elif st.session_state.page == "player_details":
    if 'show_player_details_page' in globals():
        show_player_details_page()
    else:
        st.error("Player Details page failed to load.")
elif st.session_state.page == "video_analysis":
    if 'show_video_analysis_page' in globals():
        show_video_analysis_page()
    else:
        st.error("Video Analysis page failed to load.")

# Add custom CSS (optional, kept from original)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    .stButton>button {
        width: 100%; /* Make buttons fill sidebar width */
        text-align: left;
        background-color: #262730;
        color: white;
        border-radius: 4px;
        border: 1px solid #4CAF50;
        margin-bottom: 5px; /* Add some space between buttons */
    }
    .stButton>button:hover {
        background-color: #4CAF50;
        color: white;
        border: 1px solid #ffffff;
    }
    .stButton>button:focus {
        background-color: #4CAF50; /* Keep highlight on active page */
        color: white;
        border: 1px solid #ffffff;
        box-shadow: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
