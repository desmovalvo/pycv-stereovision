#!/usr/bin/python

# requirements
from scipy.spatial import distance
from numpy.linalg import norm
from termcolor import colored
from random import randint
import numpy as np
import getopt
import time
import sys
import cv2
import pdb

def pixelbased_pixel_value(ref_pix, target_image):

    # algorithm parameters
    min_distance = 100000
    
    # initialization
    disparity = 0

    # calculate the output value
    for xx in xrange(max(x-disp_range, 0), min(x+disp_range, o_width)):
        tar_pix = target_image[y, xx]
        d = distance.euclidean(ref_pix, tar_pix) 
        if d < min_distance:
            min_distance = d
            disparity = x - xx

    # return 
    return int(float(255 * abs(disparity)) / (2 * disp_range))


def build_integral_images(height, width, img):

    # initialize the ii matrix
    ii = np.zeros((height, width, 1), dtype=np.uint8)

    # algorithm
    for y in xrange(height):
        for x in xrange(width):
            
            if (y > 0) and (x > 0):
                ii[y][x] = ii[y-1][x] + ii[y][x-1] + norm(img[y][x], ord=2) 

            elif (y > 0) and (x == 0):
                ii[y,x] = ii[y-1][x] + norm(img[y][x], ord=2)

            elif (y == 0) and (x > 0):
                ii[y][x] = ii[y][x-1] + norm(img[y][x], ord=2)
            
    # return
    return ii


def get_integral(x, y, width, height, ii, window_size):

    d = ii[min(y + window_size, height-1)][min(x + window_size, width - 1)][0]
    c = ii[min(y + window_size, height - 1)][max(0, x - window_size)][0]
    b = ii[max(0, y - window_size)][min(x + window_size, width - 1)][0]
    a = ii[max(0, y - window_size)][max(0, x - window_size)][0]

    return float(d - c - b + a)


def fixedwindow_pixel_value(ref_ii, tar_ii, ref_img, tar_img, out_img, width, height, disp_range, window_size):

    # build the output
    y = 0
    for pixel in xrange(o_height * o_width):
        
        # determine x and y
        newy = pixel / o_width
        if y != newy:
            y = newy
            print "RIGA %s" % y
        x = pixel % o_width

        # initialize disparity and distance
        min_distance = sys.maxint
        disparity = 0
        
        # aggregation for the reference pixel
        ref_sum = get_integral(x, y, width, height, ref_ii, window_size)
        print "REF sum: %s " % ref_sum

        # iterate over the pixel of the target image
        # between x-d and x+d 
        for xx in xrange(max(x-disp_range, 0), min(x+disp_range, o_width)):
            tar_sum = get_integral(xx, y, width, height, tar_ii, window_size)            
            d = np.square(tar_sum) + np.square(ref_sum) - (2 * tar_sum * ref_sum)
            if d < min_distance:
                min_distance = d
                disparity = x - xx
                print "NEW DISP: %s - NEW DIST: %s" % (disparity, min_distance)
        
        # determine the pixel value for the output image
        pixel_value = int(float(255 * abs(disparity)) / (2 * disp_range))
        out_img.itemset((y, x, 0), pixel_value)

    # return
    return out_img


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

    # algorithm parameters
    disp_range = 15

    # open images
    print colored("main> ", "blue", attrs=["bold"]) + "Opening image %s" % leftimage
    l_img = cv2.imread(leftimage)
    print colored("main> ", "blue", attrs=["bold"]) + "Image %s has resolution %sx%s" % (leftimage, l_img.shape[1], l_img.shape[0])

    print colored("main> ", "blue", attrs=["bold"]) + "Opening image %s" % rightimage
    r_img = cv2.imread(rightimage)
    print colored("main> ", "blue", attrs=["bold"]) + "Image %s has resolution %sx%s" % (rightimage, r_img.shape[1], r_img.shape[0])

    # initialization of outputimage
    print colored("main> ", "blue", attrs=["bold"]) + "Initializing output image"
    outimage = "out.png"
    o_width = max(r_img.shape[1], l_img.shape[1])
    o_height = max(r_img.shape[0], l_img.shape[0])
    size = (o_height, o_width, 1)
    o_img = np.zeros(size, dtype=np.uint8)

    # integral images
    print colored("main> ", "blue", attrs=["bold"]) + "Building integral images matrices"
    l_ii = build_integral_images(o_height, o_width, l_img)
    r_ii = build_integral_images(o_height, o_width, r_img)

    # segmentation
    # print colored("main> ", "blue", attrs=["bold"]) + "Performing segmentation on image %s" % leftimage,
    # start_time = time.clock() * 1000
    # l_s_img = cv2.pyrMeanShiftFiltering(l_img, 30, 30)
    # end_time = time.clock() * 1000
    # print "%s ms" % round(end_time - start_time, 3)

    # print colored("main> ", "blue", attrs=["bold"]) + "Performing segmentation on image %s" % rightimage,
    # start_time = time.clock() * 1000
    # r_s_img = cv2.pyrMeanShiftFiltering(r_img, 30, 1)
    # end_time = time.clock() * 1000
    # print "%s ms" % round(end_time - start_time, 3)

    print colored("main> ", "blue", attrs=["bold"]) + "Building depth map"
    out_img = fixedwindow_pixel_value(l_ii, r_ii, l_img, r_img, o_img, o_width, o_height, 15, 4)

    # y = 0
    # for pixel in xrange(o_height * o_width):
        
    #     # determine x and y
    #     newy = pixel / o_width
    #     if y != newy:
    #         y = newy
    #         print "RIGA %s" % y
    #     x = pixel % o_width

    #     # get a pixel from the reference image
    #     ref_pix = l_img[y,x]
           
    #     # determine the pixel value for the output image
    #     pv = pixelbased_pixel_value(ref_pix, r_img)
    #     o_img.itemset((y, x, 0), pv)

    # display the output image
    cv2.imshow("output image", out_img)
    cv2.waitKey(0)

    # write the image to file
    cv2.imwrite(outimage, out_img)
