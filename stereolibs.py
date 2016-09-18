#!/usr/bin/python

# system-wide requirements
import cv2
import sys
import numpy
import datetime
import ConfigParser
from math import sqrt,pow
from termcolor import colored

# local requirements
from utilities import *
from StereoException import *

###############################################################
#
# Pixel-based - the naive algorithm -- ALG 0
#
###############################################################

# pixel based
def pixelbased(ref_image, tar_image, out_image, settings_file):

    # read settings
    print colored("pixelbased> ", "blue", attrs=["bold"]) + "Reading algorithm settings"
    try:
        config = ConfigParser.ConfigParser()
        config.read(settings_file)
        settings = {}
        settings["policy"] = config.get("pixelbased", "policy")
        settings["disp_range"] = config.getint("pixelbased", "disp_range")
    except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
        raise StereoException("Missing option in configuration file!")
        
    if settings["policy"] not in ["ABS_DIF", "SQR_DIF", "TRA_DIF"]:
        raise StereoException("Policy MUST be one of ABS_DIF, SQR_DIF or TRA_DIF!")
                
    try:
        settings["threshold"] = config.getint("pixelbased", "threshold")
    except:
        if settings["policy"] == "TRA_DIF":
            raise StereoException("Missing option in configuration file!")
        else:
            settings["threshold"] = None

    # get height and width
    print colored("pixelbased> ", "blue", attrs=["bold"]) + "Reading image properties"
    width = max(out_image.shape[1], out_image.shape[1])
    height = max(out_image.shape[0], out_image.shape[0])

    # transform to bw
    print colored("pixelbased> ", "blue", attrs=["bold"]) + "Convert images to BW"
    ref_bw = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    tar_bw = cv2.cvtColor(tar_image, cv2.COLOR_BGR2GRAY)

    # iterate over the pixels
    print colored("pixelbased> ", "blue", attrs=["bold"]) + "Building disparity map"
    for pixel in xrange(height * width):
        
        # determine x and y
        y = pixel / width
        x = pixel % width

        # get a pixel from the reference image
        ref_pix = ref_bw[y,x]
           
        # per-pixel initialization
        disparity = 0
        min_distance = sys.maxint

        # calculate the output value
        for xx in xrange(max(x - settings["disp_range"], 0), x+1): #min(x + settings["disp_range"], width)):

            # retrieve the target pixel 
            tar_pix = tar_bw[y,xx]

            # matching cost
            if settings["policy"] == "ABS_DIF":
                d = abs(int(ref_pix) - int(tar_pix))

            elif settings["policy"] == "SQR_DIF":
                d = (int(tar_pix) - int(ref_pix))**2

            elif settings["policy"] == "TRA_DIF":
                d = min(abs(int(ref_pix) - int(tar_pix)), settings["threshold"])

            if d <= min_distance:
                min_distance = d
                disparity = x - xx


        # determine the pixel value for the output image
        pv = int(float(255 * abs(disparity)) / (settings["disp_range"]))
        out_image.itemset((y, x, 0), pv)

    # output file name
    out_name = get_output_filename("PIXELBASED", settings["policy"], None, settings["disp_range"], settings["threshold"], None)

    # return
    return out_image, out_name


###############################################################
#
# Fixed Window -- ALG 1
#
###############################################################

def fixedwindow(ref_image, tar_image, out_image, settings_file):

    # read settings
    print colored("fixedwindow> ", "blue", attrs=["bold"]) + "Reading algorithm settings"
    try:
        config = ConfigParser.ConfigParser()
        config.read(settings_file)
        settings = {}
        settings["disp_range"] = config.getint("fixedwindow", "disp_range")
        settings["window_size"] = config.getint("fixedwindow", "window_size")
        settings["policy"] = config.get("fixedwindow", "policy")
    except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
        raise StereoException("Missing option in configuration file!")
        
    if settings["policy"] not in ["ABS_DIF", "SQR_DIF", "TRA_DIF"]:
        raise StereoException("Policy MUST be one of ABS_DIF, SQR_DIF or TRA_DIF!")
                
    try:
        settings["threshold"] = config.getint("fixedwindow", "threshold")
    except:
        if settings["policy"] == "TRA_DIF":
            raise StereoException("Missing option in configuration file!")
        else:
            settings["threshold"] = None

    # get height and width
    print colored("fixedwindow> ", "blue", attrs=["bold"]) + "Reading image properties"
    width = max(out_image.shape[1], out_image.shape[1])
    height = max(out_image.shape[0], out_image.shape[0])

    # convert to BW
    print colored("fixedwindow> ", "blue", attrs=["bold"]) + "Converting images to BW"
    ref_bw = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    tar_bw = cv2.cvtColor(tar_image, cv2.COLOR_BGR2GRAY)

    # iterate over the pixels
    print colored("fixedwindow> ", "blue", attrs=["bold"]) + "Building disparity map"
    for pixel in xrange(height * width):

        # determine x and y
        y = pixel / width
        x = pixel % width

        # get the window value for the reference pixel
        py = range(y-settings["window_size"],y+settings["window_size"])
        px = range(x-settings["window_size"],x+settings["window_size"])
        indices = [(yyy * width + xxx) for xxx in px for yyy in py]
        pw1 = ref_bw.take(indices, mode="wrap")      
        minmatrix = numpy.full(pw1.shape, settings["threshold"])
        
        # initialize disparity and distance
        min_distance = sys.maxint
        disparity = 0

        # iterate over the pixel of the target image
        # between x-d and x+d 
        for xx in xrange(max(x-settings["disp_range"], 0), x+1):

            # initialize d
            d = 0

            # get the window value for the target pixel
            pxx = range(xx-settings["window_size"],xx+settings["window_size"])
            indices = [(yyyy * width + xxxx) for xxxx in pxx for yyyy in py]
            tw1 = tar_bw.take(indices, mode="wrap")                
            
            # matching cost
            if settings["policy"] == "ABS_DIF":
                d = numpy.sum(abs(pw1.astype(float)-tw1.astype(float)))
            
            elif settings["policy"] == "SQR_DIF":
                d = numpy.sum((pw1 - tw1)**2)

            elif settings["policy"] == "TRA_DIF":
                d = numpy.sum(numpy.minimum(abs(pw1-tw1), minmatrix))

            if d < min_distance:
                min_distance = d
                disparity = abs(x - xx)
        
        # determine the pixel value for the output image
        pixel_value = int(float(255 / settings["disp_range"]) * disparity)
        out_image.itemset((y, x, 0), pixel_value)

    # output file name
    out_name = get_output_filename("FIXEDWINDOW", settings["policy"], settings["window_size"]*2+1, settings["disp_range"], settings["threshold"], None)

    # return
    return out_image, out_name


###############################################################
#
# Intensity matrix building function
#
###############################################################

def build_intensity_matrix(img):

    """This function is used to calculate the intensity of each pixel
    for a given picture"""

    # get height and width
    width = img.shape[1]
    height = img.shape[0]

    # initialize the intensity_matrix matrix
    intensity_matrix = []

    # algorithm
    for y in xrange(height):
        
        # add a line
        intensity_matrix.append([])

        for x in xrange(width):
            intensity_matrix[y].append(sqrt(img[y][x][0]**2 + img[y][x][1]**2 + img[y][x][2]**2))

    # return
    return intensity_matrix


###############################################################
#
# Segmentation-based - ALG 2
#
###############################################################

def segmentation_based(ref_image, tar_image, out_image, settings_file):

    # read settings
    print colored("segbased> ", "blue", attrs=["bold"]) + "Reading algorithm settings"
    config = ConfigParser.ConfigParser()
    config.read(settings_file)       
    try:
        settings = {}
        settings["disp_range"] = config.getint("segbased", "disp_range")
        settings["window_size"] = config.getint("segbased", "window_size")
        settings["policy"] = config.get("segbased", "policy")
        settings["seg_threshold"] = config.getint("segbased", "seg_threshold")
    except (ConfigParser.NoSectionError, ConfigParser.NoOptionError):
        raise StereoException("Missing option in configuration file!")
        
    if settings["policy"] not in ["ABS_DIF", "SQR_DIF", "TRA_DIF"]:
        raise StereoException("Policy MUST be one of ABS_DIF, SQR_DIF or TRA_DIF!")
                
    try:
        settings["threshold"] = config.getint("segbased", "threshold")
    except:        
        if settings["policy"] == "TRA_DIF":
            raise StereoException("Missing option in configuration file!")
        else:
            settings["threshold"] = None

    # get height and width
    print colored("segbased> ", "blue", attrs=["bold"]) + "Reading image properties"
    width = out_image.shape[1]
    height = out_image.shape[0]
    
    # build the intensity matrices of reference and target images
    # the intensity matrix contains for each pixel its euclidean norm
    print colored("segbased> ", "blue", attrs=["bold"]) + "Building intensity matrices"
    ref_intensity_matrix = numpy.array(build_intensity_matrix(ref_image))
    tar_intensity_matrix = numpy.array(build_intensity_matrix(tar_image))

    # segmentation of the reference image and intensity matrix of the segmented image
    # the intensity matrix contains for each pixel its euclidean norm
    print colored("segbased> ", "blue", attrs=["bold"]) + "Segmentation of the reference image"
    ref_seg_image = cv2.pyrMeanShiftFiltering(ref_image, 30, 30)
    ref_seg_intensity_matrix = numpy.array(build_intensity_matrix(ref_image))
    
    # iterate over the pixels
    print colored("segbased> ", "blue", attrs=["bold"]) + "Building disparity map"
    for (y, x), value in numpy.ndenumerate(tar_intensity_matrix):
            
        # initialize disparity and distance
        min_distance = sys.maxint
        disparity = 0

        # calculate indices for reference image
        py = xrange(y-settings["window_size"],y+settings["window_size"]+1)
        px = xrange(x-settings["window_size"],x+settings["window_size"]+1)    

        # get the window of the reference image centered on (x,y)
        indices = [(yyy * width + xxx) for xxx in px for yyy in py]
        pw1 = ref_intensity_matrix.take(indices, mode="wrap")

        # get the window of the segmented image centered on (x,y)
        seg_window = ref_seg_intensity_matrix.take(indices, mode="wrap")
        seg_bool = (seg_window - ref_seg_intensity_matrix[y][x]) < settings["seg_threshold"]
        ref_win_sum = numpy.sum(seg_bool * pw1)
        if settings["policy"] == "TRA_DIF":
            minmatrix = numpy.full(pw1.shape, settings["threshold"])
        
        # iterate over the pixel of the target image between x-d and x 
        for xx in xrange(max(x-settings["disp_range"], 0), x+1):
                            
            # calculate indices for target image
            pxx = xrange(xx-settings["window_size"],xx+settings["window_size"]+1)
            indices = [(yyyy * width + xxxx) for xxxx in pxx for yyyy in py]

            # get the window of the target image centered on (xx,y)
            tw1 = tar_intensity_matrix.take(indices, mode="wrap")

            # matching cost
            if settings["policy"] == "ABS_DIF":
                d = numpy.sum(abs(pw1*seg_bool - tw1))
                
            elif settings["policy"] == "SQR_DIF":
                d = numpy.sum((pw1*seg_bool - tw1)**2)

            elif settings["policy"] == "TRA_DIF":
                d = numpy.sum(numpy.minimum(abs(pw1-tw1), minmatrix))
                
            # comparison
            if d < min_distance:
                min_distance = d
                disparity = x - xx

        # determine the pixel value for the output image
        pixel_value = int(float(255 * abs(disparity)) / (settings["disp_range"]))
        out_image.itemset((y, x, 0), pixel_value)
                    
    # output file name
    out_name = get_output_filename("SEGBASED", settings["policy"], settings["window_size"]*2+1, settings["disp_range"], settings["threshold"], settings["seg_threshold"])

    # return
    return out_image, out_name
