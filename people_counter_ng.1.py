#From https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/
#v2
# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi

# import the necessary packages
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')

#region Object Tracking
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
#endregion
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

#region Timeout
from utility import RepeatedTimer
if sys.version_info >= (3, 0):
	from queue import Queue
else:
	from Queue import Queue
#endregion

#region k means
from kmeans import kmeans
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from kneed import KneeLocator
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
#endregion

#region agglomerativeclustering
from agglomerativeclustering import agglomerativeclustering
#endregion

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=1,
	help="# of skip frames between detections")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
#ap.add_argument("-c", "--clustering-model", type=str, default="statisticalclustering",
#	help="enable various statistical clustering calculations")

args = vars(ap.parse_args())

#region Tracking
# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
	tracker = cv2.Tracker_create(args["tracker"].upper())

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
	# initialize a dictionary that maps strings to their corresponding
	# OpenCV object tracker implementations
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}

	# grab the appropriate object tracker using our dictionary of
	# OpenCV object tracker objects
	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
	#clusteringmodeltracker = args["clustering-model"]
	clusteringmodeltracker = "statisticalclustering"


#endregion

#region Detection
# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
#endregion

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

#region Timer thread, exit after X seconds
timerFired = None
def exitTimer(name):
    global timerFired
    timerFired = True
#rt = RepeatedTimer(10, exitTimer, "Trigger the exit timer")
#endregion

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()

#region kmeans elbow
X = np.array([1, -1])
#endregion

# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if args["input"] is not None and frame is None:
		break

	# resize the frame to have a maximum width of 500 pixels (the
	# less data we have, the faster we can process it), then convert
	# the frame from BGR to RGB for dlib
	frame = imutils.resize(frame, width=500)
	#rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	# initialize the current status along with our list of bounding
	# box rectangles returned by either (1) our object detector or
	# (2) the correlation trackers
	status = "Waiting"
	rects = []

	# initialize the bounding box coordinates of the object we are going
	# to track
	#initBB = None

	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames % args["skip_frames"] == 0:
		
		# grab the new bounding box coordinates of the object
		# resize the frame (so we can process it faster) and grab the
		# frame dimensions
		frame2 = imutils.resize(frame, width=500)
		(H, W) = frame2.shape[:2]

		# set the status and initialize our new set of object trackers
		status = "Detecting"
		trackers = []

		# convert the frame to a blob and pass the blob through the
		# network and obtain the detections
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		print (detections)

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])

				# if the class label is not a person, ignore it
				if CLASSES[idx] != "person":
					continue
				
				print (detections)
				# compute the (x, y)-coordinates of the bounding box
				# for the object
				print("detections[0, 0, i, 3:7]")
				print(detections[0, 0, i, 3:7])

				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")
				print ("box1")
				print (box)

				# start OpenCV object tracker using the supplied bounding box
				# coordinates, then start the FPS throughput estimator as well
				#tracker.init(frame, initBB)
				
				box = (startX, startY, endX, endY)
				print ("box2")
				print (box)
				tracker.init(frame2, box)

				# construct a dlib rectangle object from the bounding
				# box coordinates and then start the dlib correlation
				# tracker
				#tracker = dlib.correlation_tracker()
				#rect = dlib.rectangle(startX, startY, endX, endY)
				#tracker.start_track(rgb, rect)

				#region k means
				print ("boxy")
				boxy = np.array([startX, startY, endX, endY])
				rects.append(boxy.astype("int"))
				print (boxy)
				#endregion

				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				trackers.append(tracker)

		#print(rects)

	# otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing throughput
	else:
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"

			# update the tracker and grab the updated position
			#tracker.update(rgb)
			#pos = tracker.get_position()

			# unpack the position object
			#startX = int(pos.left())
			#startY = int(pos.top())
			#endX = int(pos.right())
			#endY = int(pos.bottom())

			#region k means
			# add the bounding box coordinates to the rectangles list
			#boxy = np.array([startX, startY, endX, endY])
			#rects.append(boxy.astype("int"))
			#endregion

			(success, box) = tracker.update(frame)
			#if success:
			#	(x, y, w, h) = [int(v) for v in box]
			#	cv2.rectangle(frame, (x, y), (x + w, y + h),
			#		(0, 255, 0), 2)

	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects)

	#region k means
	X = []
	#endregion
		
	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			#y = [c[1] for c in to.centroids]
			#direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			# check to see if the object has been counted or not
			#if not to.counted:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
				#if direction < 0 and centroid[1] < H // 2:
				#	totalUp += 1
				#	to.counted = True

				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
				#elif direction > 0 and centroid[1] > H // 2:
				#	totalDown += 1
				#	to.counted = True

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		#text = "ID {}".format(objectID)
		#cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			#cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		#region k means
		entry = [centroid[0], centroid[1]]
		X.append(entry)
		#endregion

	#region k means
	if not args.get("enable-kmeans", False):
		km = kmeans(X, cv2, frame)
		km.calculate()

		if len(X) < 4:
			skip_frames = 1
		else:
			skip_frames = 30		

	#region k means
	if not args.get("clustering-model", True):
		cm = agglomerativeclustering(rects, cv2, frame)
		cm.calculate()

	#endregion

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	#region Display
	fps.stop()
	fps.fps()
	info = [
		("Objects", len(X)),
		("Tracker", args["tracker"]),
		("FPS", "{:.2f}".format(fps.fps()))
	]

	# loop over the info tuples and draw them on our frame
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	fps = FPS().start()
	#endregion

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	if timerFired == True:
		break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()