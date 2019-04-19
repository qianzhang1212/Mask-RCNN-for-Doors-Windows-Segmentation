import cv2
import os

name = 'IMG_2617'

pred_dir = os.path.join("predictions", name)
frame = cv2.imread(os.path.join(pred_dir, "frame000.jpg"))
size = frame.shape[1], frame.shape[0]

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
vidout = cv2.VideoWriter(name + '-predictions.avi', fourcc, 29, size)

count = 0
while(1):
    frame = cv2.imread(os.path.join(pred_dir, "frame%03d.jpg" % count))
    if frame is None:
        break

    size = frame.shape[1], frame.shape[0]

    vidout.write(frame)  #write frame to the output video
    count += 1

print (count)
vidout.release()
