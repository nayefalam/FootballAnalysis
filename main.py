from utils import read_video, save_video
import numpy as np
from trackers import Tracker
from teamm_assigner import TeamAssigner
from player_ball_assigner import playerBallAssigner
from camera_movement_estimator import CameraMovementEstimator

def main():
    #read video
    video_frames = read_video('input_vid/sample_vid.mp4')

    #initialize the tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.getObjectTracks(video_frames, read_fom_stub= True, stub_path= 'stubs/track_stub1.pk1')

     # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    #Interpolate ball positions
    tracks["ball"] = tracker .interpolate_ball_pos(tracks["ball"])

    #Assigning player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']) :
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], 
                                                 track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = tuple(map(int, team_assigner.team_colors[team]))#team_assigner.team_colors[team]

    '''
    #save cropped image of a player:
    for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]

        #crop bbox from frames:
        cropped_image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        #save the cropped image
        cv2.imwrite(f'output_vid/cropped_img.jpg', cropped_image)
    '''

    #Assign Ball to a player
    playerAssigner = playerBallAssigner()

    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = playerAssigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    ####

    # Function to calculate and print total possession at the end
    def calculate_total_possession(team_ball_control):
        # Calculate the total frames where each team had the ball
        team_1_frames = np.sum(team_ball_control == 1)
        team_2_frames = np.sum(team_ball_control == 2)
        total_frames = team_1_frames + team_2_frames

        # Calculate possession percentages
        team_1_possession = (team_1_frames / total_frames) * 100
        team_2_possession = (team_2_frames / total_frames) * 100

        # Print the results in the terminal
        print(f"Total Possession - Team 1: {team_1_possession:.2f}%")
        print(f"Total Possession - Team 2: {team_2_possession:.2f}%")

    # At the end of your video processing function, call this function
    calculate_total_possession(team_ball_control)

    #draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    #save video
    save_video(output_video_frames, 'output_vid/output_vid.avi')
    #print("Team 1 color:", team_assigner.team_colors[1])
    #print("Team 2 color:", team_assigner.team_colors[2])

if __name__ == '__main__':
    main()