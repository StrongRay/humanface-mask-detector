import cv2

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import tensorflow
import numpy as np
import argparse
import os

flipCamera  = 2  # 0 or 2 - Try either
cameraNo    = 0  # can be 1 for the other camera check with ls -ltrh /dev/video* to see which number
dispWidth   = 640
dispHeight  = 480
conLevel    = 0.5

def load():
	#proto = os.path.sep.join([args["face"], "deploy.prototxt"])
	#weights = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
	net = cv2.dnn.readNet("./simple_face_detector/deploy.prototxt", "./simple_face_detector/res10_300x300_ssd_iter_140000.caffemodel")
	model = load_model("mask_detector.model")
	return model, net

def process(model, net):

    camSet='nvarguscamerasrc sensor_id='+str(cameraNo)+' !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flipCamera)+' ! video/x-raw, width='+str(dispWidth)+', height='+str(dispHeight)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
    cam= cv2.VideoCapture(camSet)
	# image = cv2.imread(args["image"])
    while True:
        ret, frame = cam.read()
        origne = frame.copy()
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0,123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if ( confidence > conLevel ):
                box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w-1, endX), min(h-1, endY))
   			    # extract face ROI, convert it to RGB channel
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224,224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                #pass the face through the model to determince if the face has a mask or not
                (mask, withoutMask) = model.predict(face)[0]
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask)*100)
                cv2.putText(frame, label, (startX, startY-10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
        cv2.imshow("Ouput", frame)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
    cam.release()
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
	print(" - [x]: Load the model from disk... ->")
	model, net = load()
	print(" - [x]: Trying to predict ... ->")
	process(model,  net)
