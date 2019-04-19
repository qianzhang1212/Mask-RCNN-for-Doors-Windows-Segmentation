#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 16:42:18 2018

@author: nat
"""

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

# val_set size
SET_SIZE = 6000

# Which weights to start with?
init_with = "house"  # imagenet, coco, house, or last

CLASS_MAP = dict([("door", 1), ("garage_door", 2), ("window", 3)])


############################################################
#  Configurations
############################################################

class InferenceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "houses"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

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
#  Utility Functions
############################################################

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    overlaps, class_names,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = utils.trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = utils.trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Loop through predictions and find matching ground truth boxes
    print (" ")
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > 0:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                print ("%s: prediction score %f - IoU %f" % (class_names[pred_class_ids[i]], pred_scores[i], iou))
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break
    print (" ")
    return gt_match, pred_match, overlaps

############################################################
#  Evaluating
############################################################
    
if __name__ == '__main__':
    inference_config = InferenceConfig()
    
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
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

    del dataset_processed
    
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = val_set.image_ids
    APs = []
    for image_id in image_ids:
        image_id = 0
        print (image_id, val_set.image_reference(image_id))
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(val_set, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=1)
        r = results[0]
        
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            val_set.class_names, r['scores'], ax=get_ax())
        visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, 
                            val_set.class_names, [1.0]*gt_class_id.shape[0], ax=get_ax())
        
        compute_matches(gt_bbox, gt_class_id, gt_mask,  
                        r["rois"], r["class_ids"], r["scores"], r['masks'],
                        overlaps, val_set.class_names)
        
        APs.append(AP)
        
    print("mAP: ", np.mean(APs))
