import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
import cv2
import multiprocessing as mp
import config as cfg

def salmap_from_norm_coords(norm_coords, sigma, height_width):
    '''
    Base function to compute general saliency maps, given the normalized (from 0 to 1)
    fixation coordinates, the sigma of the gaussian blur, and the height and
    width of the saliency map in pixels.
    '''

    img_coords = np.mod(np.round(norm_coords * np.array((height_width[1], height_width[0]))), np.array((height_width[1], height_width[0]))-1.0).astype(int)

    gaze_counts = np.zeros((height_width[0], height_width[1]))
    for coord in img_coords:
        gaze_counts[coord[1], coord[0]] += 1.0

    gaze_counts[0, 0] = 0.0
    sigma_y = sigma
    salmap = ndimage.gaussian_filter1d(gaze_counts, sigma=sigma_y, mode='wrap', axis=0)

    # In x-direction, we scale the radius of the gaussian kernel the closer we get to the pole
    for row in range(salmap.shape[0]):
        # Save the salmap as png
        angle = (row/float(salmap.shape[0]) - 0.5) * np.pi
        sigma_x = sigma_y / (np.cos(angle) + 1e-3)
        salmap[row,:] = ndimage.gaussian_filter1d(salmap[row,:], sigma=sigma_x, mode='wrap')

    # normalize
    salmap /= np.max(salmap)
    # salmap /= float(np.sum(salmap))
    return salmap

def core_function(video_names, df, OUTPUT_PATH, SALIENCY_FPS, SIGMA, HEIGHT_WIDTH, SAVE_FIXATIONS, OUTPUT_PATH_FIXATIONS):
    for video in video_names:
        
        if not os.path.exists(os.path.join(OUTPUT_PATH, str(video).zfill(4))):
            os.makedirs(os.path.join(OUTPUT_PATH, str(video).zfill(4)))
        if SAVE_FIXATIONS and not os.path.exists(OUTPUT_PATH_FIXATIONS):
            os.makedirs(OUTPUT_PATH_FIXATIONS)

        min_frame = df.loc[df['video'] == video, 'frame'].min()
        max_frame = df.loc[df['video'] == video, 'frame'].max()
        step = int(round(60/SALIENCY_FPS))

        if SAVE_FIXATIONS:
            # Create a dataframe for the fixations for the current video with the columns: video, frame, u, v, id
            df_fixations = pd.DataFrame(columns=['video', 'frame', 'u', 'v', 'id'])

        
        for frame in range(min_frame, max_frame, step):
            if os.path.exists(os.path.join(OUTPUT_PATH, str(video).zfill(4), str(video).zfill(4) + '_' + str(frame).zfill(4) + '.png')):
                continue
            
            df_frame = df.loc[(df['frame'] >= frame) & (df['frame'] < frame + step) & (df['video'] == video)]

            # Remove fixations belonging to the same frame for the same user and the same video
            df_frame = df_frame.sort_values(by=['id', 't'])
            df_frame = df_frame.reset_index(drop=True)
            # Take only one measurement (the first point) for each fixation
            df_frame = df_frame.drop(df_frame[(df_frame['fixation'] == 1) & (df_frame['fixation'].shift(+1) == 1 ) & (df_frame['id'].shift(+1) == df_frame['id'])].index)
            df_frame = df_frame.reset_index(drop=True)
            
            frame_df = df_frame.loc[(df_frame['fixation'] == True) & (df_frame['valid_fix_classification'] == True),['u', 'v']]
            
            if SAVE_FIXATIONS:
                # Create a datafrane for the fixations of the current frame with the columns: video, frame, u, v, id
                df_fix_frame = pd.DataFrame(columns=['video', 'frame', 'u', 'v', 'id'])
                df_fix_frame['u'] = frame_df['u']
                df_fix_frame['v'] = frame_df['v']
                df_fix_frame['id'] = df_frame.loc[(df_frame['fixation'] == True) & (df_frame['valid_fix_classification'] == True),['id']]
                df_fix_frame['video'] = [video] * frame_df.shape[0]
                df_fix_frame['frame'] = [frame] * frame_df.shape[0]
                # Concatenate the fixations of the current frame to the fixations of the current video
                df_fixations = pd.concat([df_fixations, df_fix_frame])

            if frame_df.shape[0] > 0:
                # Compute the saliency map
                salmap = salmap_from_norm_coords(frame_df.values, SIGMA * HEIGHT_WIDTH[1] / 360.0, HEIGHT_WIDTH)
                # Save the saliency map as png
                cv2.imwrite(os.path.join(OUTPUT_PATH, str(video).zfill(4), str(video).zfill(4) + '_' + str(frame).zfill(4) + '.png'), (salmap * 255).astype(np.uint8))
        if SAVE_FIXATIONS:
            df_fixations.to_csv(os.path.join(OUTPUT_PATH_FIXATIONS, str(video).zfill(4) + '.csv'), index=False)

PATH_TO_GAZE_CSV = cfg.gaze_data_cvs_DSAV360
OUTPUT_PATH = cfg.out_saliency_maps_DSAV360
HEIGHT_WIDTH = (cfg.resolution[1], cfg.resolution[0])
SALIENCY_FPS = cfg.n_samples
SIGMA = cfg.sigma
N_THREADS = cfg.n_threads
SAVE_FIXATIONS = cfg.save_fixations
OUTPUT_PATH_FIXATIONS = cfg.out_fixations_DSAV360

# Read all the cvs files in the folder and concatenate them
df = pd.concat([pd.read_csv(os.path.join(PATH_TO_GAZE_CSV, f)) for f in os.listdir(PATH_TO_GAZE_CSV) if f.endswith('.csv')])

# Delete the rows with frame=0, u=0.5 and v=0.5, as they are erroneous measurements
df = df.drop(df[(df['frame'] == 0) & (df['u'] == 0.5) & (df['v'] == 0.5)].index)
df = df.reset_index(drop=True)

video_names = df['video'].unique()

if N_THREADS == 1:
    core_function(video_names, df, OUTPUT_PATH, SALIENCY_FPS, SIGMA, HEIGHT_WIDTH, SAVE_FIXATIONS, OUTPUT_PATH_FIXATIONS)
else:
    # Compute the saliency masps as independent processes
    video_names = np.array_split(video_names, N_THREADS)
    pool = mp.Pool(processes=N_THREADS)
    for i in range(N_THREADS):
        pool.apply_async(core_function, args=(video_names[i], df, OUTPUT_PATH, SALIENCY_FPS, SIGMA, HEIGHT_WIDTH, SAVE_FIXATIONS, OUTPUT_PATH_FIXATIONS))
    pool.close()
    pool.join()


