#!/usr/bin/python3

import jetson.inference
import jetson.utils

import argparse
import sys
import cv2
import time

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)
        

# process frames until the user exits
while True:
    # capture the next image
    img = input.Capture()

    # render the image
    output.Render(img)

    if cv2.waitKey(1) == ord("c"):
        jetson.utils.saveImageRGBA("img" + time.time() + ".jpg", img)
        print("saved an image!")

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
