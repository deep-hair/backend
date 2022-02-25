import cv2
import numpy as np

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

face_cascade = cv2.CascadeClassifier()

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]

    cv2.imshow('Capture - Face detection', frame)

print("Reaching the remote LAN port...")
try :
    url = 'https://solihullcam.everymanbarbers.com:2083/axis-cgi/mjpg/video.cgi?resolution=320x240&dummy=1645782766093'
    #url = "https://stream-ue1-charlie.dropcam.com:443/nexus_aac/801d8e4994614aed85fe6bccf326ab19/playlist.m3u8?public=VUhpKMjswL"
    #cap = cv2.VideoCapture('rtsp://admin:SFYZEV@78.113.98.174:554/H.264')
    cap = cv2.VideoCapture(url)
except: print('error')
print("OK")


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()

out = cv2.VideoWriter(
    'output_2.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

bg = cv2.imread('background.jpg')
cv2.resize(bg, (640, 480))
#detector = cv2.SimpleBlobDetector()
while True:

	print('About to start the Read command')
	ret, frame = cap.read()
	frame = cv2.resize(frame, (640, 480))
	frame1 = frame - bg
	#gray = cv2.Canny(frame, 100, 200)
	# # using a greyscale picture, also for faster detection
	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	# keypoints = detector.detect(frame)
	#
	# print('here')
	# im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255),
	#                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	#
	# # detect people in the image
	# # returns the bounding boxes for the detected objects
	boxes, weights = hog.detectMultiScale(gray, winStride=(1, 1), scale=1.01)
	#
	boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
	#
	boxes = non_max_suppression_fast(boxes,0.80)
	for (xA, yA, xB, yB) in boxes:
	    # display the detected boxes in the colour picture
	    cv2.rectangle(frame, (xA, yA), (xB, yB),
	                  (255, 0, 0), 2)
	# print('About to show frame of Video.')
	cv2.namedWindow("preview")
	cv2.imshow("prewiew",frame)
	cv2.waitKey(1)
	#
	# print('Running..')
	out.write(frame.astype('uint8'))

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()