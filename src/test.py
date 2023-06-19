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

break_ = False

frame_index = 0

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

prev_Gesture = 0
stroke = 0
handUp = False

def CalculateDistance(point1_x, point1_y, point2_x, point2_y):
  return round(math.sqrt((abs(point1_x-point2_x)**2)+(abs(point1_y-point2_y)**2)),2)

def GetGesture():

  gesture = 0    #0-none  1-rock   2-paper   3-scrissors

  ref_distance = CalculateDistance(index3_x, index3_y, index4_x, index4_y)
  thumb_index_distance = CalculateDistance(thumb1_x, thumb1_y, index1_x, index1_y)
  baby_index_distance = CalculateDistance(baby1_x, baby1_y, index1_x, index1_y)
  baby_thumb_distance = CalculateDistance(baby1_x, baby1_y, thumb1_x, thumb1_y)

  if len(poses) == 0:
    gesture = 0
  elif baby_index_distance < 3*ref_distance:
    gesture = 1
  elif thumb_index_distance > 2*ref_distance and baby_index_distance > 3*ref_distance and baby_thumb_distance > 3*ref_distance:
    gesture = 2
  elif thumb_index_distance > 2*ref_distance and baby_index_distance > 3*ref_distance and baby_thumb_distance < 3*ref_distance:
    gesture = 3
  else:
    gesture = 0

  return gesture

def PrintValues():
  print(ref_distance)
  print(thumb_index_distance)
  print(baby_index_distance)
  print(baby_thumb_distance)

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

# process frames until the user exits
while True:
    # capture the next image
    img = input.Capture()

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=opt.overlay)

    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        for keypoint in pose.Keypoints:
            if keypoint.ID == 0 and keypoint.y < (img.height/2)-50 and handUp == False and not GetGesture() == 0:
              handUp = True
            if keypoint.ID == 0 and keypoint.y > (img.height/2)+50 and handUp == True and not GetGesture() == 0:
              handUp = False
              stroke = stroke+1
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
        if stroke == 3:
          print(GetGesture())
          break_ = False
        if stroke > 3:
          stroke = 0
    if frame_index < 25:
      if prev_Gesture == GetGesture():
        frame_index = frame_index+1
        prev_Gesture = GetGesture()
      else:
        frame_index = 0
    else:
      print(prev_Gesture)
      break_ = True

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming() or break_:
        break