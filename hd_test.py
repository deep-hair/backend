import numpy as np
import cv2
from skimage import data, filters

# Open Video
from tqdm import tqdm

cap = cv2.VideoCapture('video_test.avi')

# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=500)

# Store selected frames in an array
frames = []
for fid in tqdm(frameIds):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# Display median frame
cv2.imshow('frame', medianFrame)
cv2.imwrite('background.jpg', medianFrame)
cv2.waitKey(0)