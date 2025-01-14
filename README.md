# D-SAV360: A Dataset of Gaze Scanpaths on 360º Ambisonic Videos

Code for *“D-SAV360: A Dataset of Gaze Scanpaths on 360º Ambisonic Videos”* ([PDF](https://graphics.unizar.es/papers/Bernal-Berdun2023dsav360.pdf))


Edurne Bernal-Berdun, Daniel Martin, Sandra Malpica, Pedro J. Perez, Diego Gutierrez, Belen Masia, and Ana Serrano

**IEEE Transactions on Visualization ans Computer Graphics (ISMAR 2023)**
Visit our [website](https://graphics.unizar.es/projects/D-SAV360/) to download the dataset and supplementary material.

## Abstract
Understanding human visual behavior within virtual reality environments is crucial to fully leverage their potential. While previous research has provided rich visual data from human observers, existing gaze datasets often suffer from the absence of multimodal stimuli. Moreover, no dataset has yet gathered eye gaze trajectories (i.e., scanpaths) for dynamic content with directional ambisonic sound, which is a critical aspect of sound perception by humans. To address this gap, we introduce D-SAV360, a dataset of 4,609 head and eye scanpaths for 360º videos with first-order ambisonics. This dataset enables a more comprehensive study of multimodal interaction on visual behavior in VR environments. We analyze our collected scanpaths from a total of 87 participants viewing 85 different videos and show that various factors such as viewing mode, content type, and gender significantly impact eye movement statistics. We demonstrate the potential of D-SAV360 as a benchmarking resource for state-of-the-art attention prediction models and discuss its possible applications in further research. By providing a comprehensive dataset of eye movement data for dynamic, multimodal virtual environments, our work can facilitate future investigations of visual behavior and attention in virtual reality.


The code has been tested with:
```
matplotlib==3.7.5
numpy==1.21.6
pandas==1.4.3
scipy==1.10.1
opencv_python==4.5.4.58 
tqdm
```

## Video Frame Extraction (frames_extraction.py)
This script processes the [videos](), extracts their frames at a specified frame rate, and optionally integrates gaze data to filter the frames of interest. If a gaze data folder is specified, frame extraction will begin from the first frame containing valid gaze data. The gaze data for our dataset is available on our project page.

The following configuration variables must be defined in the `config.py` file:
- `videos_folder`: Path to the folder containing video files.
- `gaze_data_file`: Path to the folder containing gaze data CSV files (optional). Set to `None` if not used.
- `out_frames_folder`: Path to save the extracted frames.
- `frames_fps`: Desired frame rate for frame extraction.
- `frames_resolution`: Resolution for resizing the extracted frames (e.g., `(1920, 1080)`).
- `custom_names_dict`: Dictionary to map video names to custom names (optional). Set to `None` if not used.

**Output:**
For each video, a separate folder is created in the output directory (cfg.out_frames_folder). Each frame image is named using the pattern: `<video_name>_<frame_id>.png`

Folder structure:

frames/
├── video1/
│   ├── video1_0001.png
│   ├── video1_0002.png
│   └── ...
├── video2/
│   ├── video2_0001.png
│   ├── video2_0002.png
│   └── ...

## Saliency Maps Computation (compute_gt_saliencymaps_DSAV360.py)
This code is designed to generate saliency maps from eye-tracking data ([gaze data]())for the 360º videos. Saliency maps represent areas of visual attention, highlighting where users focus their gaze. The sript takes the fixation coordinates and outputs saliency maps as images, optionally saving fixation data in CSV format.

The following configuration variables must be defined in the `config.py` file:
- `gaze_data_cvs_DSAV360`: Path to the folder containing gaze CSV files.
- `out_saliency_maps_DSAV360`: Path to save the generated saliency maps.
- `resolution`: A tuple defining the resolution of the saliency maps, e.g., (width, height).
- `n_samples`: Number of frames per second for saliency map generation.
- `sigma`: Standard deviation for Gaussian blur.
- `n_threads`: Number of threads for parallel processing.
- `save_fixations`: Boolean indicating whether to save fixation data.
- `out_fixations_DSAV360`: Path to save fixation data.

**Outputs:**
- *Saliency Maps:* Saved as PNG files in the directory defined by out_saliency_maps_DSAV360. File format: `<video_id>_<frame_number>.png`
- *Fixation Data (optional):* Saved as CSV files in the directory defined by out_fixations_DSAV360.File format: `<video_id>.csv`