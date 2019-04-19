import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import _pickle as pickle
from PIL import Image, ImageDraw

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
print (ROOT_DIR)

# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, "model"))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
HOUSE_MODEL_PATH = os.path.join(ROOT_DIR, "house_full_20k_5_5.h5")
COCO_MODEL_PATH = os.path.join(os.path.join(ROOT_DIR, "model"), "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Local path to dataset information
DATASET_INFO_PATH = os.path.join(os.path.join(ROOT_DIR, "House_imgs"), "mrcnn_dataset.p")
IMAGES_DIR = os.path.join(os.path.join(ROOT_DIR, "House_imgs"), "images")

# train_set and val_set size
SET_SIZE = 6000

# Which weights to start with?
init_with = "house"  # imagenet, coco, house, or last

CLASS_MAP = dict([("door", 1), ("garage_door", 2), ("window", 3)])


############################################################
#  Configurations
############################################################

class HouseConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "houses"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    STEPS_PER_EPOCH = 11000/2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # House has 3 classes - 1:'door', 2:'garage_door', 3:'window'


############################################################
#  Dataset
############################################################

class HouseDataset(utils.Dataset):
    def load_houses(self, start_index, size, dataset_processed, all_classes):
        # Add Classes:
        for i in range(len(all_classes)):
            self.add_class("houses", i+1, all_classes[i])

        for idx in range(start_index, start_index + size):
            #print (idx)
            self.add_image("houses", image_id = dataset_processed[idx]["idx"],
                           path = os.path.join(IMAGES_DIR, dataset_processed[idx]["img_name"]),
                           width = dataset_processed[idx]["width"],
                           height = dataset_processed[idx]["height"],
                           annotations = dataset_processed[idx]["coords"],
                           classes = dataset_processed[idx]["classes"],
                           URL = dataset_processed[idx]["URL"])
    
    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "houses":
            return super(HouseDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        classes = self.image_info[image_id]["classes"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        
        for i in range(len(annotations)):
            annotation = annotations[i]
        
            class_id = self.map_source_class_id(
                "houses.{}".format(CLASS_MAP[classes[i]]))
            if class_id:
                if len(annotation) <= 2:
                    print (image_info["URL"])
                    continue

                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(HouseDataset, self).load_mask(image_id)
    
    def annToMask(self, ann, height, width):
        m = Image.new('L', (width, height), 0)
        ImageDraw.Draw(m).polygon(ann, outline=1, fill=1)
        return np.array(m)
    
    def image_reference(self, image_id):
        """Return a link to the image in the Website."""
        image_info = self.image_info[image_id]
        if image_info["source"] != "houses":
            return super(HouseDataset, self).image_reference(image_id)
        return image_info["URL"]
            

############################################################
#  Training
############################################################

if __name__ == '__main__':
    config = HouseConfig()
    config.display()
    
    
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "house":
        model.load_weights(HOUSE_MODEL_PATH, by_name=True)
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)

    
    # Create train_set(s) and val_set + training process
    with open(DATASET_INFO_PATH, "rb") as fileObject:
        dataset_processed = pickle.load(fileObject)
        all_classes = pickle.load(fileObject)

    print ("\nPROCESSED DATA size = " + str(len(dataset_processed)))
    val_set = HouseDataset()
    val_set.load_houses(0, SET_SIZE, dataset_processed, all_classes)
    val_set.prepare()
    print (val_set.num_images)
    print ("Finish Loading VAL_SET")

    train_set = HouseDataset()
    train_set.load_houses(SET_SIZE, len(dataset_processed)-SET_SIZE, dataset_processed, all_classes)
    train_set.prepare()
    print (train_set.num_images)

    model.train(train_set, val_set,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='heads')

    model.train(train_set, val_set,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='all')

    # Save weights
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
    model_path = os.path.join(MODEL_DIR, "new_trained_house.h5")
    model.keras_model.save_weights(model_path)
