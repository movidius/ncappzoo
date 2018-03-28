#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# Utilities to help visualize the output from
# Intel® Movidius™ Neural Compute Stick (NCS) 

import skimage

def draw_bounding_box( y1, x1, y2, x2, 
                       img_draw, 
                       thickness=4, 
                       color=(255, 255, 0) ):

    rr, cc = [y1, y2, y2, y1], [x1, x1, x2, x2]
    rr, cc = skimage.draw.polygon_perimeter(
                 rr, cc, shape=img_draw.shape, clip=False)

    for x in range( 0, thickness ):
        img_draw[rr-x, cc-x] = color

    return img_draw

# ==== End of file ===========================================================
