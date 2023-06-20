#!/usr/bin/python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import sys
import math
import time
import random
import keyboard

finished = False

frame_index = 0

palm_x = 0
palm_y = 0
thumb1_x = 0
thumb1_y = 0
index1_x = 0
index1_y = 0
index3_x = 0
index3_y = 0
index4_x = 0
index4_y = 0
baby1_x = 0
baby1_y = 0 

ref_distance = 0
thumb_index_distance = 0
baby_index_distance = 0
baby_thumb_distance = 0

oponnentGesture = 0
prev_Gesture = 0
stroke = 0
handUp = False


#define the function to calculate the distance between two keypoints
def CalculateDistance(point1_x, point1_y, point2_x, point2_y):
  return round(math.sqrt((abs(point1_x-point2_x)**2)+(abs(point1_y-point2_y)**2)),2)

#define the function to calculate the gesture using the finger distances
def GetGesture():

  gesture = 0    #0-none  1-rock   2-paper   3-scrissors   4-reset

  ref_distance = CalculateDistance(index3_x, index3_y, index4_x, index4_y)
  thumb_index_distance = CalculateDistance(thumb1_x, thumb1_y, index1_x, index1_y)
  baby_index_distance = CalculateDistance(baby1_x, baby1_y, index1_x, index1_y)
  baby_thumb_distance = CalculateDistance(baby1_x, baby1_y, thumb1_x, thumb1_y)

  if len(poses) == 0:
    gesture = 0
  elif thumb1_y > baby1_y:
    gesture = 4
  elif baby_index_distance < 3*ref_distance:
    gesture = 1
  elif thumb_index_distance > 2*ref_distance and baby_index_distance > 3*ref_distance and baby_thumb_distance > 3*ref_distance:
    gesture = 2
  elif thumb_index_distance > 2*ref_distance and baby_index_distance > 3*ref_distance and baby_thumb_distance < 3*ref_distance:
    gesture = 3
  else:
    gesture = 0

  return gesture

#define a funtion that chacks if the reset gesture is shown
def WaitForReset():

  reset = False

  if thumb1_y > baby1_y:
    reset_ = True
  else:
    reset_ = False

  return reset_

#define a dunction to genratate the oponents gesture
def GenerateRandomGesture():
  return random.randint(1, 3)

#define a function to print debug values
def PrintValues():
  print("detected {:d} objects in image".format(len(poses)))
  print("Referance distance: ", ref_distance)
  print("Thumb to index distance", thumb_index_distance)
  print("Baby to index distance", baby_index_distance)
  print("Baby to thumb distance", baby_thumb_distance)
  print("Gesture:", GetGesture())
  print("Stroke: ", stroke)
  print("Frame_index:", frame_index)
  print("Oponent gesture: ", oponnentGesture)
  net.PrintProfilerTimes()
  output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="empty", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

# load the input images
error_img = jetson.utils.loadImage('./source_imgs/Error.png')
rock_img = jetson.utils.loadImage('./source_imgs/Rock.png')
paper_img = jetson.utils.loadImage('./source_imgs/Paper.png')
scissors_img = jetson.utils.loadImage('./source_imgs/Scissors.png')
reset_img = jetson.utils.loadImage('./source_imgs/Reset.png')
win_img = jetson.utils.loadImage('./source_imgs/Win.png')
lose_img = jetson.utils.loadImage('./source_imgs/Lose.png')
tie_img = jetson.utils.loadImage('./source_imgs/Tie.png')
three_img = jetson.utils.loadImage('./source_imgs/Three.png')
two_img = jetson.utils.loadImage('./source_imgs/Two.png')
one_img = jetson.utils.loadImage('./source_imgs/One.png')

# process frames until the user exits
while True:
    # capture the next image
    img = input.Capture()

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=opt.overlay)


    #go through all of the poses and keypoints and save the required values
    for pose in poses:
        for keypoint in pose.Keypoints:
            if keypoint.ID == 0 and keypoint.y < (img.height/2)-30 and handUp == False and not GetGesture() == 0 and stroke < 3:
              handUp = True
            if keypoint.ID == 0 and keypoint.y > (img.height/2) and handUp == True and not GetGesture() == 0 and stroke < 3:
              handUp = False
              stroke = stroke+1
            if keypoint.ID == 0:
              palm_x = keypoint.x
              palm_y = keypoint.y
            if keypoint.ID == 1:
              thumb1_x = keypoint.x
              thumb1_y = keypoint.y
            if keypoint.ID == 5:
              index1_x = keypoint.x
              index1_y = keypoint.y
            if keypoint.ID == 7:
              index3_x = keypoint.x
              index3_y = keypoint.y
            if keypoint.ID == 8:
              index4_x = keypoint.x
              index4_y = keypoint.y
            if keypoint.ID == 17:
              baby1_x = keypoint.x
              baby1_y = keypoint.y
        if frame_index > 8 and not finished:
          frame_index = 0
          finished = True
          stroke = 0
          frame_index = 0
          oponnentGesture = GenerateRandomGesture()
        if (prev_Gesture != GetGesture() or prev_Gesture == 0) and not finished:
          frame_index = 0
          prev_Gesture = GetGesture()
        if stroke == 3 and frame_index > 0 and prev_Gesture == GetGesture() and not prev_Gesture == 0 and not finished:
          prev_Gesture = GetGesture()
          frame_index = frame_index+1
        if stroke == 3 and frame_index == 0 and not finished:
          prev_Geture = GetGesture()
          frame_index = 1

    if stroke == 0:
      jetson.utils.cudaOverlay(three_img, img, img.width/2 - three_img.width/2, img.height*0.02)
    elif stroke == 1:
      jetson.utils.cudaOverlay(two_img, img, img.width/2 - two_img.width/2, img.height*0.02)
    elif stroke == 2:
      jetson.utils.cudaOverlay(one_img, img, img.width/2 - one_img.width/2, img.height*0.02)

    if finished:
      # overlay the current gesture image in front of the hand position
      if prev_Gesture == 0:
        jetson.utils.cudaOverlay(error_img, img, palm_x - error_img.width/2, palm_y - error_img.height/2)
      elif prev_Gesture == 1:
        jetson.utils.cudaOverlay(rock_img, img, palm_x - rock_img.width/2, palm_y - rock_img.height/2)
      elif prev_Gesture == 2:
        jetson.utils.cudaOverlay(paper_img, img, palm_x - paper_img.width/2, palm_y - paper_img.height/2)
      elif prev_Gesture == 3:
        jetson.utils.cudaOverlay(scissors_img, img, palm_x - scissors_img.width/2, palm_y - scissors_img.height/2)
      elif prev_Gesture == 4:
        jetson.utils.cudaOverlay(reset_img, img, palm_x - reset_img.width/2, palm_y - reset_img.height/2)
        finished = False
        frame_index = 0
        stroke = 0

       # overlay the oponents gesture image on the side of the frame
      if oponnentGesture == 1:
        jetson.utils.cudaOverlay(rock_img, img, img.width*0.1, img.height/2)
      elif oponnentGesture == 2:
        jetson.utils.cudaOverlay(paper_img, img, img.width*0.1, img.height/2)
      elif oponnentGesture == 3:
        jetson.utils.cudaOverlay(scissors_img, img, img.width*0.1, img.height/2)

      #check for the result
      if oponnentGesture == 1 and prev_Gesture == 1:
        jetson.utils.cudaOverlay(tie_img, img, img.width/2 - tie_img.width/2, img.height/2 - tie_img.height/2)
      elif oponnentGesture == 1 and prev_Gesture == 2:
        jetson.utils.cudaOverlay(win_img, img, img.width/2 - win_img.width/2, img.height/2 - win_img.height/2)
      elif oponnentGesture == 1 and prev_Gesture == 3:
        jetson.utils.cudaOverlay(lose_img, img, img.width/2 - lose_img.width/2, img.height/2 - lose_img.height/2)
      elif oponnentGesture == 2 and prev_Gesture == 1:
        jetson.utils.cudaOverlay(lose_img, img, img.width/2 - lose_img.width/2, img.height/2 - lose_img.height/2)
      elif oponnentGesture == 2 and prev_Gesture == 2:
        jetson.utils.cudaOverlay(tie_img, img, img.width/2 - tie_img.width/2, img.height/2 - tie_img.height/2)
      elif oponnentGesture == 2 and prev_Gesture == 3:
        jetson.utils.cudaOverlay(win_img, img, img.width/2 - win_img.width/2, img.height/2 - win_img.height/2)
      elif oponnentGesture == 3 and prev_Gesture == 1:
        jetson.utils.cudaOverlay(win_img, img, img.width/2 - win_img.width/2, img.height/2 - win_img.height/2)
      elif oponnentGesture == 3 and prev_Gesture == 2:
        jetson.utils.cudaOverlay(lose_img, img, img.width/2 - lose_img.width/2, img.height/2 - lose_img.height/2)
      elif oponnentGesture == 3 and prev_Gesture == 3:
        jetson.utils.cudaOverlay(tie_img, img, img.width/2 - tie_img.width/2, img.height/2 - tie_img.height/2)

      if WaitForReset():
        finished = False

    # render the image
    output.Render(img)

    if keyboard.is_pressed("q"):
      jetson.utils.saveImageRGBA("./imgs/img" + str(time.time()) + ".jpg", img)

    # exit on input/output EOS and print the final gesture ID
    if not input.IsStreaming() or not output.IsStreaming():
        print("Final Gesture:", prev_Gesture)
        jetson.utils.saveImageRGBA("./imgs/img" + str(time.time()) + ".jpg", img)
        print("saved an image!")
        time.sleep(4)
        break