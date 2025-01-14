import cv2
import os
import config as cfg
from tqdm import tqdm
import pandas as pd


video_names = os.listdir(cfg.videos_folder)

if not cfg.gaze_data_file == None:
    df = pd.concat([pd.read_csv(os.path.join(cfg.gaze_data_file, f)) for f in os.listdir(cfg.gaze_data_file) if f.endswith('.csv')])
    # Delete the rows with frame=0, u=0.5 and v=0.5, as they are erroneous measurements
    df = df.drop(df[(df['frame'] == 0) & (df['u'] == 0.5) & (df['v'] == 0.5)].index)
    df = df.reset_index(drop=True)


with tqdm(range(len(video_names)), ascii=True) as pbar1:
    for video_name in video_names:
        print(video_name)

        video = cv2.VideoCapture(os.path.join(cfg.videos_folder, video_name, video_name +".mp4"))
        fps = video.get(cv2.CAP_PROP_FPS)

        if not cfg.gaze_data_file == None:
            init_frame = df.loc[df['video'] == int(video_name), 'frame'].min()
            last_frame = df.loc[df['video'] == int(video_name), 'frame'].max()
        else:
            init_frame = 0
            last_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        

        if fps < cfg.frames_fps:
            print("FPS of video is lower than the desired FPS. Frames will be extracted at maximum fps ({}).".format(fps))
            step = 1
        else:
            step = int(round(fps/cfg.frames_fps))

        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        new_video_folder = os.path.join(cfg.out_frames_folder, video_name.split(".")[0])
        new_video_name = video_name.split(".")[0]

        if not os.path.exists(new_video_folder):
            os.makedirs(new_video_folder)
        
        # Set the frame to the first frame
        video.set(cv2.CAP_PROP_POS_FRAMES, init_frame)

        success, frame = video.read()
        frame_id = init_frame
        frame_name = new_video_name + '_' + str(frame_id).zfill(4) + '.png'
        frame = cv2.resize(frame, cfg.frames_resolution)
        cv2.imwrite(os.path.join(new_video_folder, frame_name), frame)
        frame_id += 1
 
        with tqdm(range(length), ascii=True) as pbar2:
            while success and frame_id <= last_frame:
                success, frame = video.read()

                if (frame_id - init_frame) % step == 0 and success:
                    frame_name = new_video_name + '_' + str(frame_id).zfill(4) + '.png'
                    frame = cv2.resize(frame, cfg.frames_resolution)
                    cv2.imwrite(os.path.join(new_video_folder, frame_name), frame)
                frame_id += 1
                pbar2.update(1)

        pbar1.update(1)