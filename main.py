#!/usr/bin/python

# requirements
from scipy.spatial import distance
from numpy.linalg import norm
from termcolor import colored
from random import randint
from stereolibs import *
from math import sqrt
import numpy as np
import getopt
import time
import sys
import cv2
import pdb

# main
if __name__ == "__main__":

    # read command line parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], "vl:r:c:", ["verbose", "leftimg=", "rightimg=", "config="])
    except getopt.GetoptError as err:
        sys.exit(2)

    verbose = False
    leftimage = None
    rightimage = None
    configfile = "stereo.conf"
    for opt, arg in opts:
        if opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-l", "--leftimg"):
            leftimage = arg
        elif opt in ("-l", "--rightimg"):
            rightimage = arg
        elif opt in ("-c", "--config"):
            configfile = arg
        else:
            assert False, "unhandled option"

    # read settings
    print colored("main> ", "blue", attrs=["bold"]) + "Reading global settings"
    config = ConfigParser.ConfigParser()
    config.read(configfile)
    settings = {}
    settings["alg"] = config.get("global", "algorithm")
    settings["output_file"] = config.get("global", "output_file")

    # open images
    print colored("main> ", "blue", attrs=["bold"]) + "Opening image %s" % leftimage
    l_img = cv2.imread(leftimage)
    print colored("main> ", "blue", attrs=["bold"]) + "Image %s has resolution %sx%s" % (leftimage, l_img.shape[1], l_img.shape[0])

    print colored("main> ", "blue", attrs=["bold"]) + "Opening image %s" % rightimage
    r_img = cv2.imread(rightimage)
    print colored("main> ", "blue", attrs=["bold"]) + "Image %s has resolution %sx%s" % (rightimage, r_img.shape[1], r_img.shape[0])

    # initialization of outputimage
    print colored("main> ", "blue", attrs=["bold"]) + "Initializing output image"
    o_file = settings["output_file"]
    o_width = max(r_img.shape[1], l_img.shape[1])
    o_height = max(r_img.shape[0], l_img.shape[0])
    o_size = (o_height, o_width, 1)
    o_img = np.zeros(o_size, dtype=np.uint8)

    # build disparity map
    if settings["alg"].upper() == "PIXEL_BASED":
        out_img = pixelbased(l_img, r_img, o_img, configfile)

    elif settings["alg"].upper() == "FIXED_WINDOW":
        out_img = fixedwindow(l_img, r_img, o_img, configfile)

    elif settings["alg"].upper() == "SHIFTABLE_WINDOW":
        out_img = shiftablewindow(l_img, r_img, o_img, configfile)

    elif settings["alg"].upper() == "MULTIPLE_WINDOWS":
        out_img = multiplewindows(l_img, r_img, o_img, configfile)

    elif settings["alg"].upper() == "SEGBASED":
        out_img = segmentation_based(l_img, r_img, o_img, configfile)

    # display the output image
    cv2.imshow("output image", out_img)
    cv2.waitKey(0)

    # write the image to file
    cv2.imwrite(o_file, out_img)
