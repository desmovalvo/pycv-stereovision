#!/usr/bin/python

# system-wide requirements
import datetime

def get_output_filename(algorithm, policy, window, disparity, threshold, seg_threshold):

    # date
    d = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # algorithm
    if algorithm == "PIXELBASED":
        alg = "pixelbased"
        segthr = ""
        win = ""
    elif algorithm == "FIXEDWINDOW":
        alg = "fixedwindow"
        segthr = ""
        win = "winsize%s-" % window
    else:
        alg = "segbased"
        segthr = "-segthr%s" % seg_threshold
        win = "winsize%s-" % window

    # policy
    if policy == "ABS_DIF":
        pol = "sad"
        thr = ""
    elif policy == "SQR_DIF":
        pol = "ssd"
        thr = ""
    else:
        pol = "tad"
        thr = "-threshold%s" % threshold

    # determine output file name
    # ALGORITHM - POLICY - WINDOW - DISPARITY - THRESHOLD - SEGTHRESHOLD
    out_name = "%s-%s-%s-%sdisp%s%s%s.png" % (d, alg, pol, win, disparity, thr, segthr)
    return out_name
