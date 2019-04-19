import os
import sys
import random
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
LIB_DIR = os.path.join(ROOT_DIR, "model")
# Import Mask RCNN
sys.path.append(LIB_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
import argparse

parser = argparse.ArgumentParser(description='Visualize the targeted image')
parser.add_argument('-f', '--file_name', type=str, help='name of the targeted image in images folder')
args = parser.parse_args()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "model/logs")

# Local path to trained weights file
HOUSE_MODEL_PATH = os.path.join(ROOT_DIR, "house_full_20k_5_5.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = "houses"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load trained weights
model.load_weights(HOUSE_MODEL_PATH, by_name=True)


# Class names
class_names = ['BG', 'door', 'garage_door', 'window']


# Load a random image from the images folder
if args.file_name:
    file_name = args.file_name
else:
    file_name = random.choice(next(os.walk(IMAGE_DIR))[2])
image = skimage.io.imread(os.path.join(IMAGE_DIR, file_name))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
