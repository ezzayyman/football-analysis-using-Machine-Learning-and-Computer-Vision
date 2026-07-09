import streamlit as st
import os
import sys
import tempfile
import traceback
from pathlib import Path
import subprocess

# Add the parent directory (7uda) and the cv_pipeline directory to sys.path
APP_DIR = Path(__file__).parent.parent # Points to 7uda directory
CV_PIPELINE_DIR = APP_DIR / "cv_pipeline"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(CV_PIPELINE_DIR))

# Try importing the processing function
try:
    from cv_pipeline.main import process_video
    print("Successfully imported process_video")
except ImportError as e:
    st.error(f"Failed to import processing module: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during import: {e}")
    st.error(traceback.format_exc())
    st.stop()

def convert_avi_to_mp4(avi_path, mp4_path):
    """Converts an AVI video file to MP4 using ffmpeg."""
    try:
        command = [
            "ffmpeg",
            "-i", avi_path,
            "-c:v", "libx264", # Video codec
            "-crf", "23",       # Constant Rate Factor (quality, lower is better/larger file)
            "-preset", "fast",  # Encoding speed vs compression
            "-c:a", "aac",       # Audio codec
            "-b:a", "128k",      # Audio bitrate
            mp4_path,
            "-y" # Overwrite output file if it exists
        ]
        print(f"Running ffmpeg command: {" ".join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("FFmpeg STDOUT:", result.stdout)
        print("FFmpeg STDERR:", result.stderr)
        print(f"Successfully converted {avi_path} to {mp4_path}")
        return True
    except FileNotFoundError:
        st.error("ffmpeg not found. Please ensure ffmpeg is installed and in the system PATH.")
        print("Error: ffmpeg not found.")
        return False
    except subprocess.CalledProcessError as e:
        st.error(f"Error during video conversion: {e}")
        st.error(f"FFmpeg STDERR: {e.stderr}")
        print(f"Error converting video: {e}")
        print(f"FFmpeg STDERR: {e.stderr}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during conversion: {e}")
        print(f"Unexpected error during conversion: {e}")
        return False

def show_video_analysis_page():
    st.title("Football Video Analysis")
    st.write("Upload a video to analyze player movements, team assignments, and ball control.")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("Analyze Video"):
            with st.spinner("Analyzing video... This may take a while depending on video length."):
                try:
                    # Create temporary directory for processing
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_dir_path = Path(temp_dir)
                        input_video_path = temp_dir_path / uploaded_file.name
                        output_avi_path = temp_dir_path / "output_video.avi"
                        output_mp4_path = temp_dir_path / "output_video.mp4"
                        stubs_dir_path = temp_dir_path / "stubs"
                        reports_dir_path = temp_dir_path / "reports"
                        model_path = CV_PIPELINE_DIR / "models" / "best.pt"

                        # Ensure model file exists
                        if not model_path.exists():
                            st.error(f"Model file not found at: {model_path}")
                            st.stop()

                        # Save uploaded video to the temporary file
                        with open(input_video_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.write(f"Video saved temporarily to: {input_video_path}")

                        st.write("Starting analysis...")
                        # Call the actual analysis function
                        processed_avi_path, metrics = process_video(
                            input_video_path=str(input_video_path),
                            output_video_path=str(output_avi_path),
                            model_path=str(model_path),
                            stubs_dir=str(stubs_dir_path),
                            reports_dir=str(reports_dir_path)
                        )

                        st.success("Analysis complete!")

                        # Display results
                        st.subheader("Analysis Results")

                        # Convert AVI to MP4 for display
                        st.write("Converting output video to MP4 for display...")
                        conversion_success = convert_avi_to_mp4(str(processed_avi_path), str(output_mp4_path))

                        if conversion_success and output_mp4_path.exists():
                            st.write("Processed Video (MP4):")
                            video_file = open(output_mp4_path, 'rb')
                            video_bytes = video_file.read()
                            st.video(video_bytes)
                            video_file.close()
                            # Offer download for MP4
                            st.download_button(
                                label="Download Processed Video (MP4)",
                                data=video_bytes,
                                file_name="processed_video.mp4",
                                mime="video/mp4"
                            )
                        elif processed_avi_path and Path(processed_avi_path).exists():
                             st.warning("Could not convert video to MP4 for display. Offering AVI download instead.")
                             with open(processed_avi_path, "rb") as file:
                                st.download_button(
                                    label="Download Processed Video (AVI)",
                                    data=file,
                                    file_name="output_video.avi",
                                    mime="video/x-msvideo"
                                )
                        else:
                            st.error("Output video file not found after processing.")

                        st.write("Metrics:")
                        st.json(metrics)

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.error(traceback.format_exc())
                    print(f"Error during Streamlit analysis execution: {e}")
                    print(traceback.format_exc())

# To run this page directly for testing (optional)
# if __name__ == "__main__":
#     show_video_analysis_page()
