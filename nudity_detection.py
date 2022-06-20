''' Refs:
1) https://ourcodeworld.com/articles/read/1347/how-to-detect-nudity-nudity-detection-nsfw-content-with-machine-learning-using-nudenet-in-python
2) https://pypi.org/project/NudeNet/

Comments:
NudeNet only works with:
    TensorFlow <= 1.15.4
Therefore, it does not run with Python 3.10.
Ref 1 indicates to use Python 3.7. 
In this installation, I used Python 3.7.9.
 '''

# Getting images' path list:
import os
import inspect
from pprint import pprint

# Current (this script's) path:
script_path = os.path.dirname(
                    os.path.abspath(
                        inspect.getfile(
                            inspect.currentframe())))

# Photos' path                     
photos_folder = os.path.join(script_path, 'photos')

# List with all the photos in folder:
photos = os.listdir(photos_folder)

photos_fullpath = [os.path.join(photos_folder, photo) for photo in photos]

# Import module
from nudenet import NudeClassifier

# initialize classifier (downloads the checkpoint file automatically the first time)
classifier = NudeClassifier()

# Classify single image
# classifier.classify('path_to_image_1')
# Returns {'path_to_image_1': {'safe': PROBABILITY, 'unsafe': PROBABILITY}}

# Classify multiple images (batch prediction)
# batch_size is optional; defaults to 4
probabilities = classifier.classify(photos_fullpath, batch_size=4)
# Returns {'path_to_image_1': {'safe': PROBABILITY, 'unsafe': PROBABILITY},
#          'path_to_image_2': {'safe': PROBABILITY, 'unsafe': PROBABILITY}}
pprint(probabilities)


# Classify video
# batch_size is optional; defaults to 4
# classifier.classify_video('path_to_video', batch_size=BATCH_SIZE)
# Returns {"metadata": {"fps": FPS, "video_length": TOTAL_N_FRAMES, "video_path": 'path_to_video'},
#          "preds": {frame_i: {'safe': PROBABILITY, 'unsafe': PROBABILITY}, ....}}