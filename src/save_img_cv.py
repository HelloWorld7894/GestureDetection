#!/usr/bin/python3
import cv2
import os
import time

vid = cv2.VideoCapture(0)
  
while(True):
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    if cv2.waitKey(1) == ord("q"):
        break
    if cv2.waitKey(1) == ord("c"):
        cv2.imwrite(os.getcwd() + "/imgs/img" + str(time.time()) + ".jpg", frame)

vid.release()
cv2.destroyAllWindows()