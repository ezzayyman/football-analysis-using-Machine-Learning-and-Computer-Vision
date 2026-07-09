# Refactored main.py for Streamlit integration
import sys
import os
from pathlib import Path

# Add the cv_pipeline directory to sys.path to allow relative imports
CV_PIPELINE_DIR = Path(__file__).parent
sys.path.append(str(CV_PIPELINE_DIR))

from cv_utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
# Assuming performance metrics calculation might be too slow or complex for direct Streamlit use initially
# from performance import calculate_yolo_accuracy, plot_confusion_matrix, plot_precision_recall_f1, save_accuracy_report, plot_roc_curves
# import matplotlib.pyplot as plt
# import seaborn as sns

def process_video(input_video_path: str, output_video_path: str, model_path: str, stubs_dir: str = None, reports_dir: str = None):
    """Processes the input video to perform football analysis.

    Args:
        input_video_path: Path to the input video file.
        output_video_path: Path to save the processed output video.
        model_path: Path to the YOLO model file (e.g., best.pt).
        stubs_dir: Directory to read/write stub files (optional).
        reports_dir: Directory to save report files (optional).

    Returns:
        A tuple containing:
        - str: Path to the saved output video.
        - dict: Dictionary containing analysis metrics (e.g., team ball control).
    """
    try:
        # Ensure output directories exist
        output_dir = Path(output_video_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        if stubs_dir:
            Path(stubs_dir).mkdir(parents=True, exist_ok=True)
        if reports_dir:
            Path(reports_dir).mkdir(parents=True, exist_ok=True)

        # Read Video
        print(f"Reading video from: {input_video_path}")
        video_frames = read_video(input_video_path)
        if not video_frames:
            raise ValueError("Could not read video frames.")
        print(f"Read {len(video_frames)} frames.")

        # Initialize Tracker
        print(f"Initializing tracker with model: {model_path}")
        tracker = Tracker(model_path)

        # Determine stub paths if stubs_dir is provided
        track_stub_path = os.path.join(stubs_dir, 'track_stubs.pkl') if stubs_dir else None
        read_track_stub = bool(stubs_dir and os.path.exists(track_stub_path))

        print(f"Getting object tracks... Read from stub: {read_track_stub}")
        tracks = tracker.get_object_tracks(video_frames,
                                           read_from_stub=read_track_stub,
                                           stub_path=track_stub_path)
        print("Object tracking complete.")

        # --- Performance Metrics Calculation (Optional/Commented Out) ---
        # This section might be computationally expensive for a web app
        # accuracy_metrics = calculate_yolo_accuracy(tracks)
        # confusion_matrix_file = plot_confusion_matrix(accuracy_metrics['confusion_matrix'])
        # prf1_plot_file = plot_precision_recall_f1(accuracy_metrics)
        # roc_curves_file = plot_roc_curves(accuracy_metrics['roc_data'])
        # print("YOLO Model Accuracy Report:") # ... (print statements omitted)
        # if reports_dir:
        #     save_accuracy_report(accuracy_metrics, output_dir=reports_dir)
        # else:
        #     save_accuracy_report(accuracy_metrics) # Saves to default location
        # --------------------------------------------------------------

        print("Adding positions to tracks...")
        tracker.add_position_to_tracks(tracks)

        # Camera movement estimator
        print("Estimating camera movement...")
        camera_movement_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_stub_path = os.path.join(stubs_dir, 'camera_movement_stub.pkl') if stubs_dir else None
        read_cam_move_stub = bool(stubs_dir and os.path.exists(camera_movement_stub_path))
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                 read_from_stub=read_cam_move_stub,
                                                                                 stub_path=camera_movement_stub_path)
        camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
        print("Camera movement estimation complete.")

        # View Transformer
        print("Applying view transformation...")
        view_transformer = ViewTransformer()
        view_transformer.add_transformed_position_to_tracks(tracks)
        print("View transformation complete.")

        # Speed and distance estimator
        print("Estimating speed and distance...")
        speed_and_distance_estimator = SpeedAndDistance_Estimator()
        speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
        print("Speed and distance estimation complete.")

        # Initialize team assigner
        print("Assigning teams...")
        team_assigner = TeamAssigner()
        # Assign team colors based on a few sample frames
        sample_frames = [min(i, len(video_frames)-1) for i in range(0, min(len(video_frames), 50), 10)] # Sample frames safely
        for frame_idx in sample_frames:
            if 'players' in tracks and frame_idx < len(tracks['players']) and tracks['players'][frame_idx]:
                 team_assigner.assign_team_color(video_frames[frame_idx], tracks['players'][frame_idx])

        # Assign teams to all players across all frames
        if 'players' in tracks:
            for frame_num, player_track in enumerate(tracks['players']):
                if player_track:
                    for player_id, track_info in player_track.items():
                        team = team_assigner.get_player_team(video_frames[frame_num], track_info['bbox'], player_id)
                        tracks['players'][frame_num][player_id]['team'] = team
                        tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(team, (255, 255, 255)) # Default white
        print("Team assignment complete.")

        # Assign Ball Acquisition
        print("Assigning ball acquisition...")
        player_assigner = PlayerBallAssigner()
        team_ball_control_raw = []
        if 'players' in tracks and 'ball' in tracks:
            last_team_with_ball = 0 # 0: None, 1: Team1, 2: Team2
            for frame_num, player_track in enumerate(tracks['players']):
                current_team_has_ball = 0
                if frame_num < len(tracks['ball']) and tracks['ball'][frame_num] and 1 in tracks['ball'][frame_num]: # Check if ball exists and has track ID 1
                    ball_bbox = tracks['ball'][frame_num][1]['bbox']
                    assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

                    if assigned_player != -1 and assigned_player in tracks['players'][frame_num]:
                        tracks['players'][frame_num][assigned_player]['has_ball'] = True
                        team = tracks['players'][frame_num][assigned_player].get('team')
                        if team == "team1":
                            current_team_has_ball = 1
                        elif team == "team2":
                            current_team_has_ball = 2
                
                if current_team_has_ball != 0:
                    last_team_with_ball = current_team_has_ball
                team_ball_control_raw.append(last_team_with_ball)

        else:
            print("Warning: Player or ball tracks not found for ball acquisition assignment.")
            team_ball_control_raw = [0] * len(video_frames) # Assign no control if tracks missing

        team_ball_control = np.array(team_ball_control_raw)
        print("Ball acquisition assignment complete.")

        # Calculate metrics
        metrics = {}
        if len(team_ball_control) > 0:
            team1_frames = np.sum(team_ball_control == 1)
            team2_frames = np.sum(team_ball_control == 2)
            total_controlled_frames = team1_frames + team2_frames
            if total_controlled_frames > 0:
                metrics['team1_ball_control_percentage'] = round(team1_frames / total_controlled_frames * 100, 2)
                metrics['team2_ball_control_percentage'] = round(team2_frames / total_controlled_frames * 100, 2)
            else:
                 metrics['team1_ball_control_percentage'] = 0
                 metrics['team2_ball_control_percentage'] = 0
        else:
            metrics['team1_ball_control_percentage'] = 0
            metrics['team2_ball_control_percentage'] = 0

        # Draw Annotations
        print("Drawing annotations...")
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
        output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
        output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
        print("Annotations drawn.")

        # Save Video
        print(f"Saving output video to: {output_video_path}")
        save_video(output_video_frames, output_video_path)
        print("Output video saved.")

        return output_video_path, metrics

    except Exception as e:
        print(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise the exception so Streamlit can catch it
        raise

# Keep the main block for potential standalone testing if needed
# if __name__ == '__main__':
#     # Example usage for standalone testing
#     # Create dummy input/output paths for testing
#     test_input = 'path/to/your/test/video.mp4'
#     test_output = 'path/to/your/output/video.avi'
#     test_model = 'path/to/your/model/best.pt'
#     test_stubs = 'path/to/your/stubs'
#     test_reports = 'path/to/your/reports'
#     if os.path.exists(test_input) and os.path.exists(test_model):
#         process_video(test_input, test_output, test_model, stubs_dir=test_stubs, reports_dir=test_reports)
#     else:
#         print("Please set valid paths for test_input and test_model in the __main__ block.")

