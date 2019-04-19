# View Analysis on Doors, Garage Doors, and Windows by Instance Segmentation using Region Proposals

### This repository includes:
* The instance segmentation model is based on Mask R-CNN ([paper](https://arxiv.org/abs/1703.06870) + [github](https://github.com/matterport/Mask_RCNN))
* [House_imgs/processing_data.py](House_imgs/processing_data.py)
    * get images and data from .csv in House_imgs/Dataset folder
    * rotate and save images
    * save a dataset in a form of a list of python dict(index, img_name, width, height, coordinates, class, URL)
* [training_house.py](training_house.py)
    * train the dataset generated from processing_data.py
    * use the first 6,000 images as a validation set
* [house.py](house.py) / [house.ipynb](house.ipynb)
    * test and visualize on a random image in the folder images

### To do segmentation on an image with this model:
1. download trained weight for the instance segmentation model [weight](https://github.com/songwongtp/Mask_RCNN_ViewAnalysis/releases/tag/v0.2)
2. place the trained weight in the main directory
3. place the image to be segmented in the images folder in the main directory
4. run house.py to visualize the segmentation on an image
    * if no argument 'python house.py', an image will be randomly selected from the images folder
    * otherwise add an image name as an argument 'python house.py -f image_name' to specify the image to be detected

### Video Prediction
1. Go to video directory
2. put a video in the directory
3. change 'name' variable in video2frames.py to the name of the video (with a file type)
4. change 'name' variable in detection.py and frames2video.py to the name of the video (without a file type)
