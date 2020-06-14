Contributing to the exploration.

Added a customised code to run off GPU based NVIDIA Xavier NX and off a webcamera
Given NVIDIA has GPU, removed codes checking for CPU.

Also, **import cv2** MUST be added at the top of the code before calleing **import tensorflow** as it will give an error if CV2 is loaded after tensorflow.

Given that the training dataset was augmented with a virtual WHITE mask, the detector doesn't perform too badly.

![alt text](https://github.com/StrongRay/humanface-mask-detector/blob/master/HMD-Black-Mask.png)
![alt text](https://github.com/StrongRay/humanface-mask-detector/blob/master/HMD-No-Mask.png)
![alt text](https://github.com/StrongRay/humanface-mask-detector/blob/master/HMD-White-Mask.png)

Speed wise, its still slow.   Not sure if the **cv2.dnn.blobFromImage** is slowing or **detections = net.forward()** is slow. Only way to examine this is to put in timestamps. Or it is using a caffe model that is slow. Whatever it is, this hardware is not the issue.  

Maybe one can try YOLOV4 and see if transfer learning via using a trained model there is much faster.

Any predictions that cannot be done in near realtime and in 20-30 fps range, is virtually academic in nature.  A good detection model must be near real time. 
