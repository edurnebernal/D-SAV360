# ==================================================================================================
# Saliency map extraction parameters (compute_gt_sliencymaps_DSAV360.py):
# ==================================================================================================

# Common parameters
sigma = 5 # degrees
n_threads = 42 # number of cores to use (set to 1 to use a single thread)
n_samples = 8 # number of samples per second (fps)
resolution = (320, 240) # resolution of the saliency maps (W,H)
save_fixations = True # save the fixations in a csv file

# D-SAV360 parameters
gaze_data_cvs_DSAV360 = "./dataset/gaze_data"
out_saliency_maps_DSAV360 = "./DSAV360data/saliency_maps_"+ str(n_samples) + "fps" # output folder to store the saliency maps
out_fixations_DSAV360 = "./DSAV360data/fixations_" + str(n_samples) + "fps"# output folder to store the fixations csv files


# ==================================================================================================
# Frames extraction parameters (Benchmark_audiovisual/extract_frames.py):
# ==================================================================================================

frames_fps = 8 # frames per second
gaze_data_file = gaze_data_cvs_DSAV360 # path to the gaze data folder (None if is not DSAV360 dataset)
videos_folder = "./dataset/videos" # path to the videos folder
out_frames_folder = "./DSAV360data/frames_8fps_HR" # output folder to store the frames
frames_resolution = (3840, 3840//2) # resolution of the frames (W,H)
