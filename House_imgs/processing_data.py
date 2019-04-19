import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFile
import requests
import io
import functools
import json
from os import listdir, mkdir
from os.path import isfile, join, exists
import _pickle as pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

def rot90(point):
    """
    Rotate a point 90 degree clockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = (0,0)
    px, py = point

    qx = ox + math.cos(-math.pi/2) * (px - ox) - math.sin(-math.pi/2) * (py - oy)
    qy = oy + math.sin(-math.pi/2) * (px - ox) + math.cos(-math.pi/2) * (py - oy)
    return qx, qy

def coords_transpose(coords, orientation, H, W):
    rotated_coords = []
    for coord in coords:
        points = []
        for x,y in coord:
            if orientation == 2:
                x = -1*(x - float(W)/2) + float(W)/2

            elif orientation == 3:
                x = x - float(W)/2
                y = y - float(H)/2
                
                x,y = rot90((x,y))
                x,y = rot90((x,y))
                
                x = x + float(W)/2
                y = y + float(H)/2

            elif orientation == 4:
                y = -1*(y - float(H)/2) + float(H)/2

            elif orientation == 5:
                x = -1*(x - float(W)/2) + float(W)/2
                
                x = x - float(W)/2
                y = y - float(H)/2
                
                x,y = rot90((x,y))
                
                x = x + float(H)/2
                y = y + float(W)/2

            elif orientation == 6:
                x = x - float(W)/2
                y = y - float(H)/2
                
                x,y = rot90((x,y))
                x,y = rot90((x,y))
                x,y = rot90((x,y))
                
                x = x + float(H)/2
                y = y + float(W)/2

            elif orientation == 7:
                y = -1*(y - float(H)/2) + float(H)/2
                
                x = x - float(W)/2
                y = y - float(H)/2
                
                x,y = rot90((x,y))
                
                x = x + float(H)/2
                y = y + float(W)/2

            elif orientation == 8:
                x = x - float(W)/2
                y = y - float(H)/2
                
                x,y = rot90((x,y))
                
                x = x + float(H)/2
                y = y + float(W)/2
                
            points.append((x,y))
        rotated_coords.append(points)
        
    return rotated_coords
            

def image_transpose_exif(im, coords, H, W):
    """
        Apply Image.transpose to ensure 0th row of pixels is at the visual
        top of the image, and 0th column is the visual left-hand side.
        Return the original image if unable to determine the orientation.

        As per CIPA DC-008-2012, the orientation field contains an integer,
        1 through 8. Other values are reserved.
    """

    exif_orientation_tag = 0x0112
    exif_transpose_sequences = [                   # Val  0th row  0th col
        [],                                        #  0    (reserved)
        [],                                        #  1   top      left
        [Image.FLIP_LEFT_RIGHT],                   #  2   top      right
        [Image.ROTATE_180],                        #  3   bottom   right
        [Image.FLIP_TOP_BOTTOM],                   #  4   bottom   left
        [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],  #  5   left     top
        [Image.ROTATE_270],                        #  6   right    top
        [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],  #  7   right    bottom
        [Image.ROTATE_90],                         #  8   left     bottom
    ]

    try:
        orientation = im._getexif()[exif_orientation_tag]
        seq = exif_transpose_sequences[orientation]
    except Exception:
        return im, coords
    else:
        rotated_coords = coords_transpose(coords, orientation, H, W)
        rotated_im = functools.reduce(type(im).transpose, seq, im)
        
        return rotated_im, rotated_coords

def create_masks(H, W, coords):
    masks = []

    print ('number of objects = ' + str(len(coords)))
    for points in coords:
        temp = Image.new('L', (W, H), 0)
        ImageDraw.Draw(temp).polygon(points, outline=1, fill=1)
        masks.append(np.array(temp))
        
    return masks
    
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

mypath = './Dataset'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()

#
# {'url': ..., 'masks': ...}
#

images_dic = dict()
file_types = set()
all_classes = set()

for csv in onlyfiles:
    print (join(mypath, csv))
    df = pd.read_csv(join(mypath, csv))
    
    idx = 0
    while(True):
        column_name = 'Input.imgUrl'
        answer_coor = 'Answer.coordinates'
        if idx > 0:
            column_name = column_name + str(idx+1)
        answer_coor = answer_coor + str(idx)
        
        if column_name not in list(df):
            break
        
        #print (column_name, answer_coor)
            
        for i in range(len(df[column_name])):
            imgUrl = df[column_name][i]    
            if (imgUrl.split('.')[-1]).lower() == 'mov':
                continue
                
            file_types.add((imgUrl.split('.')[-1]).lower())
            
            contours = json.loads((df[answer_coor][i] if (pd.notnull(df[answer_coor][i]) and df[answer_coor][i] != '{}') else '[]'))
            
            if imgUrl in images_dic:
                if len(contours) > len(images_dic[imgUrl]):
                    images_dic[imgUrl] = contours
            else:
                images_dic[imgUrl] = contours
            
            for contour in contours:
                all_classes.add(contour['type'])
        idx += 1


dataset_raw = [(k,v) for k,v in images_dic.items()]
dataset_raw.sort()
all_classes = list(all_classes)
all_classes.sort()

#with open('dataset_and_classes.p', "wb") as fileObject:
#    pickle.dump(dataset_raw, fileObject)
#    pickle.dump(all_classes, fileObject)

#print (len(dataset_raw))
#print (all_classes)
#print (file_types)

#with open('dataset_and_classes.p', "rb") as f_in:
#    dataset_raw = pickle.load(f_in)
#    all_classes = pickle.load(f_in)


SET_SIZE = 250
dataset_raw = dataset_raw[:250]
print ("\n TEMPORARY SAVE at every " + str(SET_SIZE) + " images.\n")
    
dataset_processed = []
for i in range(len(dataset_raw)):
    Url, contours = dataset_raw[i]
    coords = []
    classes = []

    print (("%5d: " + Url)%(i))

    fd = requests.get(Url, stream=True)
    image_file = io.BytesIO(fd.raw.read())
    img = Image.open(image_file)

    for contour in contours:
        points = []
        for pt in contour['points']:
            points.append((pt['x'], pt['y'])) 
        coords.append(points)

        classes.append(contour['type'])
    
    W, H = img.size
    img, coords = image_transpose_exif(img, coords, H, W)

    batch_num = join('images', Url.split('/')[-2])
    if not exists(batch_num):
        mkdir(batch_num)
    
    img_name = join(Url.split('/')[-2], Url.split('/')[-1])
    W, H = img.size
    img.save('images/' + img_name)

    dataset_processed.append(dict([("idx", i), 
                                   ("img_name", img_name), 
                                   ("width", W), 
                                   ("height", H), 
                                   ("coords", coords), 
                                   ("classes", classes), 
                                   ("URL", Url)]))
    
    if ((i+1) % SET_SIZE == 0):
        print ("\nTEMPORARY SAVE at i = " + str(i+1))
        with open('mrcnn_dataset.p', "wb") as f_out:
            pickle.dump(dataset_processed, f_out)
            pickle.dump(all_classes, f_out)
        print ("SAVE COMPLETE at i = " + str(i+1) + "\n")

print ("\nFINAL SAVE")
with open('mrcnn_dataset.p', "wb") as f_out:
    pickle.dump(dataset_processed, f_out)
    pickle.dump(all_classes, f_out)
print ("SAVE COMPLETE all data\n")

# [ 
# {'image': ..., 'masks': (H, W, #instances), 'classes': (#instances, ), 'coords': (#instances, #coordinates)} 
# ]

"""
dataset_processed = []
idx = 1

out_name = 'training_set_'
count = 0
set_size = 250

for Url, contours in dataset_raw.items():
    instance = dict()
    
    print (str(idx) + ': ' + Url)
    fd = requests.get(Url, stream=True)
    image_file = io.BytesIO(fd.raw.read())
    img = Image.open(image_file)
    
    coords = []
    classes = []
    for contour in contours:
        points = []
        for pt in contour['points']:
            points.append((pt['x'], pt['y'])) 
        coords.append(points)
        
        classes.append(contour['type'])

    W, H = img.size
    img, coords = image_transpose_exif(img, coords, H, W)
    
    W, H = img.size
    masks = create_masks(H, W, coords)
    
    instance['image'] = Url.split('/')[-1]
    instance['masks'] = np.stack(masks, axis=-1)
    instance['classes'] = classes
    instance['coords'] = coords
    dataset_processed.append(instance)
    
    print (instance['masks'].shape)
    
    img.save('images/' + instance['image'])
    if idx % set_size == 0:
        print ("Saving..." _ str(count))
        with open('trains/' + out_name + str(count) + '.p', "wb") as fileObject:
            pickle.dump(dataset_processed, fileObject)
            del dataset_processed[:]
        count += 1
        print ("Finish...")

    idx += 1

    #im = np.array(img)
    #for mask in masks:
    #    im_mask = apply_mask(im, mask, (1,0,0))

    #display_image_in_actual_size(im_mask)
    #break

if len(dataset_processed) > 0:
    print ("Saving..." + str(count))
    with open('trains/' + out_name + str(count) + '.p', "wb") as fileObject:
        pickle.dump(dataset_processed, fileObject)
    print("Finish...")
"""
