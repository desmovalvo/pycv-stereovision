#!/usr/bin/python

# requirements
from termcolor import colored
import getopt
import sys
import cv2

# main
if __name__ == "__main__":

    # read command line parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], "vl:r:", ["verbose", "leftimg=", "rightimg="])
    except getopt.GetoptError as err:
        sys.exit(2)

    verbose = False
    leftimage = None
    rightimage = None
    for opt, arg in opts:
        if opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-l", "--leftimg"):
            leftimage = arg
        elif opt in ("-l", "--rightimg"):
            rightimage = arg
        else:
            assert False, "unhandled option"

    # create an instance of the image
    print colored("main> ", "blue", attrs=["bold"]) + "Opening images"
    l_img = cv2.imread(leftimage)
    r_img = cv2.imread(rightimage)
    
    # print image width and height
    
