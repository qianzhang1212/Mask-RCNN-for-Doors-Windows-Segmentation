import os
import sys
import random
import time
import colorsys
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Root directory of the project
ROOT_DIR = os.path.abspath("..")
print (ROOT_DIR)

# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, "model"))  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "model/_logs")

# Local path to trained weights file
weight_path = "house_full_20k_5_5.h5"
HOUSE_MODEL_PATH = os.path.join(ROOT_DIR, weight_path)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "video/frames")
name = 'IMG_2617'

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

class InferenceConfig(HouseConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

############################################################
#  Utility Functions
############################################################

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def save_image(image, image_name, boxes, masks, class_ids, scores, class_names,
               filter_classs_names=None, scores_thresh=0.1, save_dir=None, mode=0):
    """
        image: image array
        image_name: image name
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [num_instances, height, width]
        class_ids: [num_instances]
        scores: confidence scores for each box
        class_names: list of class names of the dataset
        filter_classs_names: (optional) list of class names we want to draw
        scores_thresh: (optional) threshold of confidence scores
        save_dir: (optional) the path to store image
        mode: (optional) select the result which you want
                mode = 0 , save image with bbox,class_name,score and mask;
                mode = 1 , save image with bbox,class_name and score;
                mode = 2 , save image with class_name,score and mask;
                mode = 3 , save mask with black background;
    """
    mode_list = [0, 1, 2, 3]
    assert mode in mode_list, "mode's value should in mode_list %s" % str(mode_list)

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "output")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    useful_mask_indices = []

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances in image %s to draw *** \n" % (image_name))

        if mode != 3:
            masked_image = image.astype(np.uint8).copy()
        else:
            masked_image = np.zeros(image.shape).astype(np.uint8)

        masked_image = Image.fromarray(masked_image)
        masked_image.save(os.path.join(save_dir, image_name))
        return
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i in range(N):
        # filter
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        if score is None or score < scores_thresh:
            continue

        label = class_names[class_id]
        if (filter_classs_names is not None) and (label not in filter_classs_names):
            continue

        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        useful_mask_indices.append(i)

    if len(useful_mask_indices) == 0:
        print("\n*** No instances in image %s to draw *** \n" % (image_name))

        if mode != 3:
            masked_image = image.astype(np.uint8).copy()
        else:
            masked_image = np.zeros(image.shape).astype(np.uint8)

        masked_image = Image.fromarray(masked_image)
        masked_image.save(os.path.join(save_dir, image_name))
        return

    #colors = random_colors(len(useful_mask_indices))
    colors = [(1.0, 0.0, 0.0) for i in range(len(useful_mask_indices))]

    if mode != 3:
        masked_image = image.astype(np.uint8).copy()
    else:
        masked_image = np.zeros(image.shape).astype(np.uint8)

    if mode != 1:
        for index, value in enumerate(useful_mask_indices):
            masked_image = apply_mask(masked_image, masks[:, :, value], colors[index])

    masked_image = Image.fromarray(masked_image)

    if mode == 3:
        masked_image.save(os.path.join(save_dir, image_name))
        return

    draw = ImageDraw.Draw(masked_image)
    colors = np.array(colors).astype(int) * 255

    for index, value in enumerate(useful_mask_indices):
        class_id = class_ids[value]
        score = scores[value]
        label = class_names[class_id]

        y1, x1, y2, x2 = boxes[value]
        if mode != 2:
            color = tuple(colors[index])
            draw.rectangle((x1, y1, x2, y2), outline=color)

        # Label
        font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf', 11)
        draw.text((x1, y1), "%s %.3f" % (label, score), (255, 255, 255), font)

    masked_image.save(os.path.join(save_dir, image_name))

############################################################
#  Evaluating
############################################################

start_time = time.time()

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(HOUSE_MODEL_PATH, by_name=True)

class_names = ['BG', 'door', 'garage_door', 'window']

# Load a random image from the images folder
file_names = next(os.walk(os.path.join(IMAGE_DIR, name)))[2]
file_names.sort()
count = len(file_names)

print (count)
print ("Preparing time = " + str(time.time() - start_time))

predictions = []
for i in range(count):
    start_time = time.time()
    if file_names[i] == '.DS_Store':
        continue
    
    img_name = file_names[i].split('.')[0]
    image = skimage.io.imread(os.path.join(os.path.join(IMAGE_DIR, name), img_name + '.jpg'))
    load_time = time.time()
    print ("Loading Image time = " + str(load_time - start_time))

    # Run detection
    results = model.detect([image], verbose=0)
    detect_time = time.time()
    print ("Detecting time = " + str(detect_time - load_time))

    # Visualize results
    predictions.append((img_name, results[0]))
    
    """
    r = results[0]
    save_image(image, img_name + '.jpg',
               r["rois"], r["masks"], r["class_ids"], r["scores"], class_names,
               save_dir=os.path.join('predictions', name + '_'))
    save_time = time.time()
    print ("Saving Result time = " + str(save_time - detect_time))
    print ("--------------------------------------------")
    """

import copy
preds = copy.deepcopy(predictions)
for i in range(len(preds)):
    img_name, r = preds[i]
    print (img_name)
    # Find arbitrary appear
    delete = []
    for j in range(len(r['scores'])):
        # If high confident skip
        if r['scores'][j] >= 0.8:            
            continue
        box = r['rois'][j:j+1]
        class_id = r['class_ids'][j:j+1]
        score = r['scores'][j:j+1]
        mask = r['masks'][:,:,j:j+1]
        
        found = False
        # Check appear before
        for k in range(1, 6):
            if i - k < 0:
                break
            _, _r = preds[i-k]
            gt_match, _, _ = utils.compute_matches(box, class_id, mask,
                                                   _r['rois'], _r['class_ids'], _r['scores'], _r['masks'],
                                                   iou_threshold=0.1, score_threshold=0.0)
            
            if gt_match[0] != -1 and _r['scores'][int(gt_match[0])] > 0.7:
                found = True
                break
        
        # Check appear after
        for k in range(1, 6):
            if i + k >= len(preds):
                break
            _, _r = preds[i+k]
            gt_match, _, _ = utils.compute_matches(box, class_id, mask,
                                                   _r['rois'], _r['class_ids'], _r['scores'], _r['masks'],
                                                   iou_threshold=0.1, score_threshold=0.0)
            if gt_match[0] != -1 and _r['scores'][int(gt_match[0])] > 0.7:
                found = True
                break
        
        # Not appear in any
        if not found:
            delete.append(j)
    
    for idx in reversed(delete):
        r['rois'] = np.delete(r['rois'], idx, 0)
        r['masks'] = np.delete(r['masks'], idx, -1)
        r['class_ids'] = np.delete(r['class_ids'], idx, 0)
        r['scores'] = np.delete(r['scores'], idx, 0)
    
    if i + 2 >= len(preds):
        continue
    
    # Find arbitrary disappear
    for j in range(len(r['scores'])):
        box = r['rois'][j:j+1]
        class_id = r['class_ids'][j:j+1]
        score = r['scores'][j:j+1]
        mask = r['masks'][:,:,j:j+1]
        
        _, _r = preds[i+1]
        gt_match, _, _ = utils.compute_matches(box, class_id, mask,
                                               _r['rois'], _r['class_ids'], _r['scores'], _r['masks'],
                                               iou_threshold=0.1, score_threshold=0.0)
        
        if gt_match[0] != -1:
            continue
        
        _, _r = preds[i+2]
        gt_match, _, _ = utils.compute_matches(box, class_id, mask,
                                               _r['rois'], _r['class_ids'], _r['scores'], _r['masks'],
                                               iou_threshold=0.1, score_threshold=0.0)
            
        if gt_match[0] != -1 and _r['scores'][int(gt_match[0])] > 0.7:
            preds[i+1][1]['rois'] = np.concatenate((preds[i+1][1]['rois'], box), axis=0)
            preds[i+1][1]['masks'] = np.concatenate((preds[i+1][1]['masks'], mask), axis=-1)
            preds[i+1][1]['class_ids'] = np.concatenate((preds[i+1][1]['class_ids'], class_id), axis=0)
            preds[i+1][1]['scores'] = np.concatenate((preds[i+1][1]['scores'], score), axis=0)
    
    image = skimage.io.imread(os.path.join(os.path.join(IMAGE_DIR, name), img_name + '.jpg'))
    save_image(image, img_name + '.jpg',
           r["rois"], r["masks"], r["class_ids"], r["scores"], class_names,
           save_dir=os.path.join('predictions', name))

