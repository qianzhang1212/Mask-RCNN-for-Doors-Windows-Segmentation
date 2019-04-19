import cv2
import os
print(cv2.__version__)

name = 'IMG_2617.MOV'

img_dir = os.path.join("frames", name.split('.')[0])
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

vidcap = cv2.VideoCapture(name)
fps = vidcap.get(cv2.CAP_PROP_FPS)
print (fps)

success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite(os.path.join(img_dir, "frame%03d.jpg" % count), image)     # save frame as JPEG file
  success,image = vidcap.read()
  #print ('Read a new frame: ', success)
  count += 1
vidcap.release()
print ('count: ', count)
