#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# Utilities to help visualize the output from
# Intel® Movidius™ Neural Compute Stick (NCS) 

import numpy
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

def draw_bounding_box( y1, x1, y2, x2, 
                       img, 
                       thickness=4, 
                       color=(255, 255, 0),
                       display_str=() ):

    """ Inputs
    (x1, y1)  = Top left corner of the bounding box
    (x2, y2)  = Bottom right corner of the bounding box
    img       = Image/frame represented as numpy array
    thickness = Thickness of the bounding box's outline
    color     = Color of the bounding box's outline
    """

    img = PIL.Image.fromarray( img )
    draw = PIL.ImageDraw.Draw( img )

    for x in range( 0, thickness ):
        draw.rectangle( [(x1-x, y1-x), (x2-x, y2-x)], outline=color )

    font = PIL.ImageFont.load_default()
    draw.text( (x1, y1), display_str, font=font )

    return numpy.array( img )

# ==== End of file ===========================================================
