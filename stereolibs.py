#!/usr/bin/python

import cv2
import pdb
import sys
import time
import numpy
import ConfigParser
from math import sqrt
from termcolor import colored

###############################################################
#
# Pixel-based - the naive algorithm -- ALG 0
#
###############################################################

# pixel based
def pixelbased(ref_image, tar_image, out_image, settings_file):

    # read settings
    print colored("pixelbased> ", "blue", attrs=["bold"]) + "Reading algorithm settings"
    config = ConfigParser.ConfigParser()
    config.read(settings_file)
    settings = {}
    settings["policy"] = config.get("pixelbased", "policy")
    settings["disp_range"] = config.getint("pixelbased", "disp_range")
    try:
        settings["threshold"] = config.getint("pixelbased", "threshold")
    except:
        pass

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

    # return
    return out_image


###############################################################
#
# Integral Images - optimization function
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


def build_integral_image_matrix(img, squared=False):

    """This function is used to calculate the integral image
    for a given picture"""

    # get height and width
    width = img.shape[1]
    height = img.shape[0]

    # initialize the ii matrix
    ii = []

    # algorithm
    for y in xrange(height):
        
        # add a line
        ii.append([])

        for x in xrange(width):

            ii[y].append(None)
            
            # euclidean norm
            # norm = sqrt(img[y][x][0]**2 + img[y][x][1]**2 + img[y][x][2]**2)

            # RGB Luminance value
            if squared:
                norm = (img[y][x][0]*0.11 + img[y][x][1]*0.59 + img[y][x][2]*0.3)**2
            else:
                norm = img[y][x][0]*0.11 + img[y][x][1]*0.59 + img[y][x][2]*0.3

            if (y > 0) and (x > 0):
                ii[y][x] = ii[y-1][x] + ii[y][x-1] - ii[y-1][x-1] + norm

            elif (y > 0) and (x == 0):
                ii[y][x] = ii[y-1][x] + norm

            elif (y == 0) and (x > 0):
                ii[y][x] = ii[y][x-1] + norm
                
            else:
                ii[y][x] = norm

    # return
    return ii


def get_integral(x, y, width, height, ii, window_size, xoffset=0, yoffset=0):

    """This function is used to get the result of the integral
    for a given window"""

    d = ii[min(y + yoffset + window_size, height-1)][min(x + xoffset + window_size, width - 1)]
    c = ii[min(y + yoffset + window_size, height - 1)][max(0, x + xoffset - window_size)]
    b = ii[max(0, y + yoffset - window_size)][min(x + xoffset + window_size, width - 1)]
    a = ii[max(0, y + yoffset - window_size)][max(0, x + xoffset - window_size)]

    return d - c - b + a


def build_integral_bw_image_matrix(img, squared=False):

    """This function is used to calculate the integral image
    for a given picture"""

    # get height and width
    width = img.shape[1]
    height = img.shape[0]

    # initialize the ii matrix
    ii = []

    # algorithm
    for y in xrange(height):
        
        # add a line
        ii.append([])

        for x in xrange(width):

            ii[y].append(None)
            
            # euclidean norm
            # norm = sqrt(img[y][x][0]**2 + img[y][x][1]**2 + img[y][x][2]**2)

            # RGB Luminance value
            if squared:
                norm = img[y][x]
            else:
                norm = img[y][x] ** 2
                            
            if (y > 0) and (x > 0):
                ii[y][x] = ii[y-1][x] + ii[y][x-1] - ii[y-1][x-1] + int(norm)

            elif (y > 0) and (x == 0):
                ii[y][x] = ii[y-1][x] + int(norm)

            elif (y == 0) and (x > 0):
                ii[y][x] = ii[y][x-1] + int(norm)
                
            else:
                ii[y][x] = norm

    # return
    return ii


###############################################################
#
# Fixed Window -- ALG 1
#
###############################################################

def fixedwindow(ref_image, tar_image, out_image, settings_file):

    # read settings
    print colored("fixedwindow> ", "blue", attrs=["bold"]) + "Reading algorithm settings"
    config = ConfigParser.ConfigParser()
    config.read(settings_file)
    settings = {}
    settings["disp_range"] = config.getint("fixedwindow", "disp_range")
    settings["window_size"] = config.getint("fixedwindow", "window_size")
    settings["policy"] = config.get("fixedwindow", "policy")
    try:
        settings["threshold"] = config.getint("fixedwindow", "threshold")
    except:
        pass

    # get height and width
    print colored("fixedwindow> ", "blue", attrs=["bold"]) + "Reading image properties"
    width = max(out_image.shape[1], out_image.shape[1])
    height = max(out_image.shape[0], out_image.shape[0])

    # convert to BW
    ref_bw = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    tar_bw = cv2.cvtColor(tar_image, cv2.COLOR_BGR2GRAY)

    # build the intensity matrices
    print colored("fixedwindow> ", "blue", attrs=["bold"]) + "Building intensity matrices"
    ref_in = numpy.array(build_intensity_matrix(ref_image))
    tar_in = numpy.array(build_intensity_matrix(tar_image))

    # build the integral images matrices
    print colored("fixedwindow> ", "blue", attrs=["bold"]) + "Building integral images matrices"
    ref_ii = build_integral_bw_image_matrix(ref_bw, True)
    tar_ii = build_integral_bw_image_matrix(tar_bw, True)

    # iterate over the pixels
    print colored("fixedwindow> ", "blue", attrs=["bold"]) + "Building disparity map"
    y = 0
    for pixel in xrange(height * width):

        # determine x and y
        y = pixel / width
        x = pixel % width

        # get the window value for the reference pixel
        py = range(y-settings["window_size"],y+settings["window_size"])
        px = range(x-settings["window_size"],x+settings["window_size"])
        # py = range(max(0, y-settings["window_size"]),min(y+settings["window_size"], height))
        # px = range(max(0, x-settings["window_size"]),min(x+settings["window_size"], width))        
        indices = [(yyy * width + xxx) for xxx in px for yyy in py]
        pw1 = ref_in.take(indices, mode="wrap")      
        minmatrix = numpy.full(pw1.shape, settings["threshold"])
        ref_sum = get_integral(x, y, width, height, ref_ii, settings["window_size"])
        
        # initialize disparity and distance
        min_distance = sys.maxint
        disparity = 0

        # iterate over the pixel of the target image
        # between x-d and x+d 
        for xx in xrange(max(x-settings["disp_range"], 0), x+1):
            
            d = 0

            # get the window value for the reference pixel
            pxx = range(xx-settings["window_size"],xx+settings["window_size"])
            # pxx = range(max(0, xx-settings["window_size"]),min(xx+settings["window_size"], width))
            indices = [(yyyy * width + xxxx) for xxxx in pxx for yyyy in py]
            tw1 = tar_in.take(indices, mode="wrap")                
            
            # matching cost
            if settings["policy"] == "ABS_DIF":
                try:
                    d = numpy.sum(abs(pw1.astype(float)-tw1.astype(float)))
                except:
                    pass
            
            elif settings["policy"] == "SQR_DIF":
                try:
                    # tar_sum = get_integral(xx, y, width, height, tar_ii, settings["window_size"])                    
                    # d = ref_sum + tar_sum - 2 * (numpy.add.reduce(pw1.astype(numpy.float) * tw1.astype(numpy.float)))
                    d = numpy.sum((pw1 - tw1)**2)
                except Exception as e: 
                    pass

            elif settings["policy"] == "TRA_DIF":
                try:
                    d = numpy.sum(numpy.minimum(abs(pw1-tw1), minmatrix))
                except:
                    pass

            if d < min_distance:
                min_distance = d
                disparity = abs(x - xx)
        
        # determine the pixel value for the output image
        # pixel_value = int(float(255 * abs(disparity)) / (settings["disp_range"]))
        pixel_value = int(float(255) / settings["disp_range"] * disparity)
        out_image.itemset((y, x, 0), pixel_value)

    # return
    return out_image


def fixedwindow_backup(ref_image, tar_image, out_image, settings_file):

    # read settings
    print colored("fixedwindow> ", "blue", attrs=["bold"]) + "Reading algorithm settings"
    config = ConfigParser.ConfigParser()
    config.read(settings_file)
    settings = {}
    settings["disp_range"] = config.getint("fixedwindow", "disp_range")
    settings["window_size"] = config.getint("fixedwindow", "window_size")
    settings["policy"] = config.get("fixedwindow", "policy")
    try:
        settings["threshold"] = config.getint("fixedwindow", "threshold")
    except:
        pass

    # get height and width
    print colored("fixedwindow> ", "blue", attrs=["bold"]) + "Reading image properties"
    width = max(out_image.shape[1], out_image.shape[1])
    height = max(out_image.shape[0], out_image.shape[0])

    # convert to BW
    ref_bw = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    tar_bw = cv2.cvtColor(tar_image, cv2.COLOR_BGR2GRAY)

    # build the integral images matrices
    print colored("fixedwindow> ", "blue", attrs=["bold"]) + "Building integral images matrices"
    ref_ii = build_integral_bw_image_matrix(ref_bw)
    tar_ii = build_integral_bw_image_matrix(tar_bw)

    # iterate over the pixels
    print colored("fixedwindow> ", "blue", attrs=["bold"]) + "Building disparity map"
    y = 0
    for pixel in xrange(height * width):

        # determine x and y
        y = pixel / width
        x = pixel % width
        
        # initialize disparity and distance
        min_distance = sys.maxint
        disparity = 0
        
        # aggregation for the reference pixel
        ref_sum = get_integral(x, y, width, height, ref_ii, settings["window_size"])

        # iterate over the pixel of the target image
        # between x-d and x+d 
        for xx in xrange(max(x-settings["disp_range"], 0), x+1): # min(x+settings["disp_range"], width)):
            
            d = 0
            tar_sum = get_integral(xx, y, width, height, tar_ii, settings["window_size"])                    

            # matching cost
            if settings["policy"] == "ABS_DIF":
                d = abs(tar_sum - ref_sum)
            elif settings["policy"] == "SQR_DIF":
                d = (tar_sum - ref_sum) ** 2
            elif settings["policy"] == "TRA_DIF":
                d = min(abs(tar_sum - ref_sum), settings["threshold"])

            if d < min_distance:
                min_distance = d
                disparity = abs(x - xx)
        
        # determine the pixel value for the output image
        # pixel_value = int(float(255 * abs(disparity)) / (settings["disp_range"]))
        pixel_value = int(float(255) / settings["disp_range"] * disparity)
        out_image.itemset((y, x, 0), pixel_value)

    # return
    return out_image



###############################################################
#
# Shiftable windows -- ALG 2
#
###############################################################

def shiftablewindow(ref_image, tar_image, out_image, settings_file):

    # read settings
    print colored("shiftablewindow> ", "blue", attrs=["bold"]) + "Reading algorithm settings"
    config = ConfigParser.ConfigParser()
    config.read(settings_file)
    settings = {}
    settings["disp_range"] = config.getint("shiftablewindow", "disp_range")
    settings["window_size"] = config.getint("shiftablewindow", "window_size")
    settings["policy"] = config.get("shiftablewindow", "policy")
    try:
        settings["threshold"] = config.getint("shiftablewindow", "threshold")
    except:
        pass

    # get height and width
    print colored("shiftablewindow> ", "blue", attrs=["bold"]) + "Reading image properties"
    width = max(out_image.shape[1], out_image.shape[1])
    height = max(out_image.shape[0], out_image.shape[0])

    # generating couples of xoffset-yoffset
    offsets = [[0,0]]

    # N S W E
    offsets.append([0, settings["window_size"]])
    offsets.append([0, -1 * settings["window_size"]])
    offsets.append([settings["window_size"], 0])
    offsets.append([-1 * settings["window_size"], 0])
    offsets.append([1, 0])
    offsets.append([0, 1])
    offsets.append([-1, 0])
    offsets.append([0, -1])

    # NW NE SW SE
    offsets.append([-1 * settings["window_size"], -1 * settings["window_size"]])
    offsets.append([-1 * settings["window_size"], settings["window_size"]])
    offsets.append([settings["window_size"], settings["window_size"]])
    offsets.append([settings["window_size"], -1 * settings["window_size"]])
    offsets.append([1, 1])
    offsets.append([-1, -1])
    offsets.append([-1, 1])
    offsets.append([1, -1])

    # build the integral images matrices
    print colored("shiftablewindow> ", "blue", attrs=["bold"]) + "Building integral images matrices"
    ref_ii = build_integral_image_matrix(ref_image)
    tar_ii = build_integral_image_matrix(tar_image)

    # iterate over the pixels
    print colored("shiftablewindow> ", "blue", attrs=["bold"]) + "Building disparity map"
    y = 0
    for pixel in xrange(height * width):

        # determine x and y
        y = pixel / width
        x = pixel % width

        # initialize disparity and distance
        min_distance = sys.maxint
        disparity = 0

        # iterate over the offsets
        for off in offsets:
            
            # aggregation for the reference pixel
            ref_sum = get_integral(x, y, width, height, ref_ii, settings["window_size"], off[0], off[1])
                
            # iterate over the pixel of the target image
            # between x-d and x+d 
            for xx in xrange(max(x-settings["disp_range"], 0), min(x+settings["disp_range"], width)):
                
                d = 0
                tar_sum = get_integral(xx, y, width, height, tar_ii, settings["window_size"], off[0], off[1])                    
    
                # matching cost
                if settings["policy"] == "ABS_DIF":
                    d = abs(tar_sum - ref_sum)
                elif settings["policy"] == "SQR_DIF":
                    d = (tar_sum - ref_sum) ** 2
                elif settings["policy"] == "TRA_DIF":
                    d = min(abs(tar_sum - ref_sum), settings["threshold"])
    
                if d < min_distance:
                    min_distance = d
                    disparity = x - xx
                    
            # determine the pixel value for the output image
            pixel_value = int(float(255 * abs(disparity)) / (settings["disp_range"]))
            out_image.itemset((y, x, 0), pixel_value)
    
    # return
    return out_image


###############################################################
#
# Multiple windows -- ALG 3
#
###############################################################

def multiplewindows(ref_image, tar_image, out_image, settings_file):

    # read settings
    print colored("multiplewindow> ", "blue", attrs=["bold"]) + "Reading algorithm settings"
    config = ConfigParser.ConfigParser()
    config.read(settings_file)
    settings = {}
    settings["disp_range"] = config.getint("multiplewindows", "disp_range")
    settings["window_size"] = config.getint("multiplewindows", "window_size")
    settings["policy"] = config.get("multiplewindows", "policy")
    try:
        settings["threshold"] = config.getint("multiplewindows", "threshold")
    except:
        pass

    # get height and width
    print colored("multiplewindow> ", "blue", attrs=["bold"]) + "Reading image properties"
    width = max(out_image.shape[1], out_image.shape[1])
    height = max(out_image.shape[0], out_image.shape[0])

    # build the integral images matrices
    print colored("multiplewindow> ", "blue", attrs=["bold"]) + "Building integral images matrices"
    ref_ii = build_integral_image_matrix(ref_image)
    tar_ii = build_integral_image_matrix(tar_image)

    # iterate over the pixels
    print colored("multiplewindow> ", "blue", attrs=["bold"]) + "Building disparity map"
    y = 0
    for pixel in xrange(height * width):

        # determine x and y
        y = pixel / width
        x = pixel % width

        # calculate the 9 sub-windows
        ref_windows = []
        
        # N S W E
        ref_windows.append(get_integral(x, y, width, height, ref_ii, settings["window_size"], settings["window_size"] * 2 + 1, 0))
        ref_windows.append(get_integral(x, y, width, height, ref_ii, settings["window_size"], settings["window_size"] * -2 + 1, 0))
        ref_windows.append(get_integral(x, y, width, height, ref_ii, settings["window_size"], 0, settings["window_size"] * 2 + 1))
        ref_windows.append(get_integral(x, y, width, height, ref_ii, settings["window_size"], 0, settings["window_size"] * -2 + 1))

        # NW NE SW SE
        ref_windows.append(get_integral(x, y, width, height, ref_ii, settings["window_size"], settings["window_size"] * 2 + 1, settings["window_size"] * 2 + 1))
        ref_windows.append(get_integral(x, y, width, height, ref_ii, settings["window_size"], settings["window_size"] * -2 + 1, settings["window_size"] * 2 + 1))
        ref_windows.append(get_integral(x, y, width, height, ref_ii, settings["window_size"], settings["window_size"] * 2 + 1, settings["window_size"] * -2 + 1))
        ref_windows.append(get_integral(x, y, width, height, ref_ii, settings["window_size"], settings["window_size"] * -2 + 1, settings["window_size"] * -2 + 1))

        # initialize disparity and distance
        min_distance = sys.maxint
        disparity = 0
            
        # aggregation for the reference pixel
        ref_sum = get_integral(x, y, width, height, ref_ii, settings["window_size"])
            
        # iterate over the pixel of the target image
        # between x-d and x+d 
        for xx in xrange(max(x-settings["disp_range"], 0), min(x+settings["disp_range"], width)):
            
            d = 0

            # aggregation for the target pixel
            tar_sum = get_integral(xx, y, width, height, tar_ii, settings["window_size"])

            # calculate the 9 sub-windows
            tar_windows = []
            
            # N S W E
            tar_windows.append(get_integral(x, y, width, height, tar_ii, settings["window_size"], settings["window_size"] * 2 + 1, 0))
            tar_windows.append(get_integral(x, y, width, height, tar_ii, settings["window_size"], settings["window_size"] * -2 + 1, 0))
            tar_windows.append(get_integral(x, y, width, height, tar_ii, settings["window_size"], 0, settings["window_size"] * 2 + 1))
            tar_windows.append(get_integral(x, y, width, height, tar_ii, settings["window_size"], 0, settings["window_size"] * -2 + 1))
            
            # NW NE SW SE
            tar_windows.append(get_integral(x, y, width, height, tar_ii, settings["window_size"], settings["window_size"] * 2 + 1, settings["window_size"] * 2 + 1))
            tar_windows.append(get_integral(x, y, width, height, tar_ii, settings["window_size"], settings["window_size"] * -2 + 1, settings["window_size"] * 2 + 1))
            tar_windows.append(get_integral(x, y, width, height, tar_ii, settings["window_size"], settings["window_size"] * 2 + 1, settings["window_size"] * -2 + 1))
            tar_windows.append(get_integral(x, y, width, height, tar_ii, settings["window_size"], settings["window_size"] * -2 + 1, settings["window_size"] * -2 + 1))

            # matching cost
            if settings["policy"] == "ABS_DIF":

                # select the best 4 additional windows
                full_list = []
                for x in xrange(len(tar_windows)):
                    full_list.append(abs(tar_windows[x] - ref_windows[x]))
                full_list.sort()
                    
                # perform the sum
                d = 0
                for x in xrange(4):
                    d += full_list[x]

            elif settings["policy"] == "SQR_DIF":

                # select the best 4 additional windows
                full_list = []
                for x in xrange(len(tar_windows)):
                    full_list.append(tar_windows[x]**2 - ref_windows[x]**2)
                full_list.sort()
                
                # perform the sum
                d = 0
                for x in xrange(4):
                    d += full_list[x]

            elif settings["policy"] == "TRA_DIF":

                # select the best 4 additional windows
                full_list = []
                for x in xrange(len(tar_windows)):
                    full_list.append(min(abs(tar_windows[x] - ref_windows[x]), settings["threshold"]))
                full_list.sort()
                
                # perform the sum
                d = 0
                for x in xrange(4):
                    d += full_list[x]

            if d < min_distance:
                min_distance = d
                disparity = x - xx
                
        # determine the pixel value for the output image
        pixel_value = int(float(255 * abs(disparity)) / (settings["disp_range"]))
        out_image.itemset((y, x, 0), pixel_value)
    
    # return
    return out_image


###############################################################
#
# Segmentation
#
###############################################################

def segment(img):
    
    """This function is used to segment an image"""
    
    # segmentation
    seg_img = cv2.pyrMeanShiftFiltering(img, 30, 30)

    # return
    return seg_img


def segmentation_based(ref_image, tar_image, out_image, settings_file):

    # read settings
    print colored("segbased> ", "blue", attrs=["bold"]) + "Reading algorithm settings"
    config = ConfigParser.ConfigParser()
    config.read(settings_file)
    settings = {}
    settings["disp_range"] = config.getint("segbased", "disp_range")
    settings["window_size"] = config.getint("segbased", "window_size")
    settings["policy"] = config.get("segbased", "policy")
    try:
        settings["threshold"] = config.getint("segbased", "threshold")
    except:
        pass

    # get height and width
    print colored("segbased> ", "blue", attrs=["bold"]) + "Reading image properties"
    width = out_image.shape[1]
    height = out_image.shape[0]
    
    # build the intensity matrices
    # TODO: multithread
    print colored("segbased> ", "blue", attrs=["bold"]) + "Building intensity matrices"
    ref_intensity_matrix = numpy.array(build_intensity_matrix(ref_image))
    tar_intensity_matrix = numpy.array(build_intensity_matrix(tar_image))

    # segmentation of the reference image
    # we also build a matrix with the euclidean norm of each pixel
    print colored("segbased> ", "blue", attrs=["bold"]) + "Segmentation of the reference image"
    ref_seg_image = segment(ref_image)
    ref_seg_intensity_matrix = numpy.array(build_intensity_matrix(ref_image))
    
    # iterate over the pixels
    print colored("segbased> ", "blue", attrs=["bold"]) + "Building disparity map"
    y = 0
    starttime = time.time() * 1000
    for x in range(width):
        for y in range(height):

            # initialize disparity and distance
            min_distance = sys.maxint
            disparity = 0

            # calculate indices for reference image
            py = range(max(0, y-settings["window_size"]),min(y+settings["window_size"], height))
            px = range(max(0, x-settings["window_size"]),min(x+settings["window_size"], width))    

            # get the window of the reference image centered on (x,y)
            indices = [(yyy * width + xxx) for xxx in px for yyy in py]
            pw1 = ref_intensity_matrix.take(indices).reshape(len(px),len(py)) #.reshape(settings["window_size"]*2, settings["window_size"]*2)

            # get the window of the segmented image centered on (x,y)
            # TODO: optimize while moving to the next column
            seg_window = ref_seg_intensity_matrix.take(indices).reshape(len(px),len(py)) #.reshape(settings["window_size"]*2, settings["window_size"]*2)
            seg_bool = (seg_window - ref_seg_intensity_matrix[y][x]) < 100
            ref_win_sum = numpy.sum(seg_bool * pw1)

            # initialize indices for target image
            
            # iterate over the pixel of the target image between x-d and x 
            tw1 = None
            for xx in xrange(max(x-settings["disp_range"], 0), x+1):

                d = 0

                # calculate indices for target image
                pxx = xrange(max(0, xx-settings["window_size"]),min(xx+settings["window_size"], width))
                indices = [(yyyy * width + xxxx) for xxxx in pxx for yyyy in py]
                
                # get the window of the target image centered on (xx,y)
                tw1 = tar_intensity_matrix.take(indices).reshape(len(pxx),len(py))
                try:
                    d = numpy.sum(abs(pw1*seg_bool - tw1))
                except:
                    pass
                    
                # comparison
                if d < min_distance:
                    min_distance = d
                    disparity = x - xx

            # determine the pixel value for the output image
            pixel_value = int(float(255 * abs(disparity)) / (settings["disp_range"]))
            out_image.itemset((y, x, 0), pixel_value)

    print time.time()*1000 - starttime
    
    # return
    return out_image
