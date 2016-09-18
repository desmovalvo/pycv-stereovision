#!/usr/bin/python

# system-wide requirements
from scipy.spatial import distance
from numpy.linalg import norm
from termcolor import colored
from random import randint
from math import sqrt
import numpy as np
import getopt
import sys
import cv2
import os

# local requirements
from stereolibs import *
from StereoException import *


# main
if __name__ == "__main__":

    # read command line parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], "l:r:c:", ["leftimg=", "rightimg=", "config="])
    except getopt.GetoptError as err:
        print colored("main> ", "red", attrs=["bold"]) + str(err)
        sys.exit(2)

    leftimage = None
    rightimage = None
    configfile = "stereo.conf"
    for opt, arg in opts:
        if opt in ("-l", "--leftimg"):
            leftimage = arg
        elif opt in ("-l", "--rightimg"):
            rightimage = arg
        elif opt in ("-c", "--config"):
            configfile = arg          

    # read settings from the configuration file
    print colored("main> ", "blue", attrs=["bold"]) + "Reading global settings"
    try:
        config = ConfigParser.ConfigParser()
        config.read(configfile)
        settings = {}
        settings["alg"] = config.get("global", "algorithm")
    except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
        print colored("main> ", "red", attrs=["bold"]) + "Missing option in configuration file!"
        sys.exit()
        
    if settings["alg"] not in ["PIXEL_BASED", "FIXED_WINDOW", "SEGBASED"]:
        print colored("main> ", "red", attrs=["bold"]) + "Algorithm MUST be one of PIXEL_BASED, FIXED_WINDOW, SEGBASED!"
        sys.exit()

    # open images
    print colored("main> ", "blue", attrs=["bold"]) + "Opening image %s" % leftimage
    if os.path.exists(leftimage):
        l_img = cv2.imread(leftimage)
    else:
        print colored("main> ", "red", attrs=["bold"]) + "Image %s not found!" % leftimage
        sys.exit()
    print colored("main> ", "blue", attrs=["bold"]) + "Image %s has resolution %sx%s" % (leftimage, l_img.shape[1], l_img.shape[0])

    print colored("main> ", "blue", attrs=["bold"]) + "Opening image %s" % rightimage
    if os.path.exists(rightimage):
        r_img = cv2.imread(rightimage)
    else:
        print colored("main> ", "red", attrs=["bold"]) + "Image %s not found!" % rightimage
        sys.exit()
    print colored("main> ", "blue", attrs=["bold"]) + "Image %s has resolution %sx%s" % (rightimage, r_img.shape[1], r_img.shape[0])

    # initialization of outputimage
    print colored("main> ", "blue", attrs=["bold"]) + "Initializing output image"
    o_width = max(r_img.shape[1], l_img.shape[1])
    o_height = max(r_img.shape[0], l_img.shape[0])
    o_size = (o_height, o_width, 1)
    o_img = np.zeros(o_size, dtype=np.uint8)

    # build disparity map
    alg = None
    if settings["alg"].upper() == "PIXEL_BASED":
        alg = pixelbased
    elif settings["alg"].upper() == "FIXED_WINDOW":
        alg = fixedwindow
    elif settings["alg"].upper() == "SEGBASED":
        alg = segmentation_based
    try:
        out_img, out_name = alg(l_img, r_img, o_img, configfile)
    except StereoException as e:
        print colored("main> ", "red", attrs=["bold"]) + "Algorithm failed! " + str(e)
        sys.exit()

    # display the output image
    cv2.imshow("output image", out_img)
    cv2.waitKey(0)

    # write the image to file
    cv2.imwrite(out_name, out_img)
