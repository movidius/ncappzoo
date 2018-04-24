import cv2
import numpy as np
from mvnc import mvncapi as mvnc
import numpy
import cv2
import os
import sys
from typing import List
import digitdetector


class TouchCalc:
    NETWORK_IMAGE_DIMENSIONS = (28, 28)

    def __init__(self, window_title='TouchCalc', width=1400, height=500):
        # Save these for use later
        self._height = height
        self._width = width

        # Create a window with OpenCV
        self._window_name = 'touch_window'
        self._window = cv2.namedWindow(self._window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self._window_name, self._width, self._height)
        cv2.setWindowTitle(self._window_name, window_title)

        # Set a flag to know when the user is drawing and assign a mouse event listener
        self._drawing = False
        self._last_point = None  # track the last point drawn for drawing lines
        cv2.setMouseCallback(self._window_name, self._draw)

        # The bottom 10% of the window will be a menu that can't be drawn on
        self._menu_bar_threshold = int(self._height * .9)

        # Set up a blank canvas
        self._canvas = np.zeros((height, width, 3), np.uint8)

        self._box_color = (230, 230, 230)

        self._operation_width = 80
        self._operation_height = 80
        self._pad = 10

        self._operand1_left = 20
        self._operand1_top = 50
        self._operand1_width = 400
        self._operand1_height = 350

        self._operand2_left = self._operand1_left + self._operand1_width + self._operation_width + self._pad + self._pad
        self._operand2_top = self._operand1_top
        self._operand2_width = self._operand1_width
        self._operand2_height = self._operand1_height

        # operation rectangle, +, -, *, / etc
        self._operation_color = (0, 0, 0)
        self._operation_left = self._operand1_left + self._operand1_width + self._pad
        self._operation_top = self._operand1_top + (int)(self._operand1_height / 2) - (int)(self._operation_height / 2)

        # equals sign
        self._equals_height = self._operation_height
        self._equals_width = self._operation_width
        self._equals_left = self._operand2_left + self._operand2_width + self._pad
        self._equals_top = self._operand2_top + (int)(self._operand2_height / 2) - (int)(self._equals_height / 2)

        self._answer_str = ""

        self._clear()

        self._device, self._graph = self.do_mvnc_initialize()

        self._infer_count = 0

    def _clear(self):
        """Clear the canvas and redraw buttons."""
        self._canvas[:] = 255
        self._answer_str = ""
        self._draw_ui()

    def _draw(self, event, x, y, flags, param):
        """Event listener for mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if y > self._menu_bar_threshold:
                if x < 100:
                    # Bottom left corner was clicked
                    self._clear()
                if x > self._width - 100:
                    # Bottom right corner was clicked
                    self.submit()
            else:
                self._drawing = True

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            if y < self._menu_bar_threshold:
                #cv2.circle(self._canvas, (x, y), 10, (0, 0, 0), -1)
                if self._last_point:
                    cv2.line(self._canvas, self._last_point, (x, y), (0, 0, 0), 10)
                    self._last_point = (x, y)
                else:
                    self._last_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            self._last_point = None

    def _draw_ui(self, color=(255, 0, 0)):
        """Draw buttons on the canvas."""
        font_name = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2
        font_thickness = 3
        y_coord = self._height - 10

        cv2.putText(self._canvas, 'C', (0, y_coord), font_name, font_scale, color, font_thickness)
        cv2.putText(self._canvas, '=', (self._width - 50, y_coord), font_name, font_scale, color, font_thickness)

        #operand 1 rect
        cv2.rectangle(self._canvas, (self._operand1_left, self._operand1_top),
                      (self._operand1_left + self._operand1_width, self._operand1_top+self._operand1_height),
                      self._box_color, 1)

        #operand 2 rect
        cv2.rectangle(self._canvas, (self._operand2_left, self._operand2_top),
                      (self._operand2_left + self._operand2_width, self._operand2_top+self._operand2_height),
                      self._box_color, 1)

        # operation ( + )
        #cv2.rectangle(self._canvas, (operation_left, operation_top), (operation_left + operation_width, operation_top+operation_height), box_color, 1)
        cv2.line(self._canvas, (self._operation_left + (int)(self._operation_width/2), self._operation_top ),
                 (self._operation_left + (int)(self._operation_width / 2), self._operation_top + self._operation_height),
                 self._operation_color, 5)
        cv2.line(self._canvas, (self._operation_left, self._operation_top + (int)(self._operation_height/2) ),
                 (self._operation_left+self._operation_width, self._operation_top + (int)(self._operation_height / 2)),
                 self._operation_color, 5)


        #cv2.rectangle(self._canvas, (equals_left, equals_top), (equals_left + equals_width, equals_top+equals_height), box_color, 1)
        cv2.line(self._canvas, (self._equals_left, self._equals_top + (int)(self._equals_height/2) - 10 ),
                 (self._equals_left+self._equals_width, self._equals_top + (int)(self._equals_height / 2) - 10),
                 self._operation_color, 5)
        cv2.line(self._canvas, (self._equals_left, self._equals_top + (int)(self._equals_height/2) + 10),
                 (self._equals_left+self._equals_width, self._equals_top + (int)(self._equals_height / 2)+ 10),
                 self._operation_color, 5)

        answer_height = 80
        answer_width = 100
        answer_left = self._equals_left + self._equals_width + self._pad
        answer_top = self._equals_top
        answer_bottom = answer_top + answer_height

        if (self._answer_str != ""):
            answer_font_scale = 4
            cv2.putText(self._canvas, self._answer_str, (answer_left, answer_bottom), font_name, answer_font_scale, color, font_thickness)
        else:
            cv2.rectangle(self._canvas, (answer_left, answer_top), (answer_left + answer_width, answer_top+answer_height), (255, 255, 255), cv2.FILLED  )


    def close(self):
        """Close and destroy the window."""
        cv2.destroyWindow(self._window_name)
        self.do_mvnc_cleanup(self._device, self._graph)

    def is_window_closed(self):
        """ Determines if the user closed the window"""
        # may only work with opencv 3.x
        # check if the window has been closed.  all properties will return -1.0
        # for windows that are closed. If the user has closed the window via the
        # x on the title bar the property will be < 0 or an exception raised.  We are
        # getting property aspect ratio but it could probably be any property

        prop_asp = 1
        try:
            prop_asp = cv2.getWindowProperty(self._window_name, cv2.WND_PROP_ASPECT_RATIO)
        except:
            #print("Caught exception, calling getWindowProperty aspect ratio")
            return True

        if (prop_asp < 0.0):
            # the property returned was < 0 so assume window was closed by user
            #print("aspect ratio is less than 0.")
            return True

        try:
            tmp = cv2.getWindowProperty(self._window_name, cv2.WND_PROP_FULLSCREEN)
        except:
            #print("Caught exception, calling getWindowProperty fullscreen")
            return True

        return False


    def show(self):
        """Show the window if hidden and update the display."""
        cv2.imshow(self._window_name, self._canvas)

    def submit(self):
        """Process the image when the submit button is clicked."""
        # Remove the buttons
        self._draw_ui(color=(255, 255, 255))

        # Detect the digits
        #digits = digitdetector.detect(self._canvas)
        #for box in digits :
        #    print(box)

        #box = digits[0]
        #print(box)

        op_pad = 40

        op1_image = self._canvas[self._operand1_top : self._operand1_top + self._operand1_height,  self._operand1_left : self._operand1_left + self._operand1_width]
        digits_op1 = digitdetector.detect(op1_image)
        print(digits_op1[0])
        op1_x1 = digits_op1[0][0]
        op1_y1 = digits_op1[0][1]
        op1_x2 = op1_x1+digits_op1[0][2]
        op1_y2 = op1_y1+digits_op1[0][3]
        op1_x1 -= op_pad
        op1_x2 += op_pad
        op1_y1 -= op_pad
        op1_y2 += op_pad
        op1_width = op1_x2 - op1_x1
        op1_height = op1_y2 - op1_y1
        if  (op1_width > op1_height):
            # wider than high
            diff = op1_width - op1_height
            op1_y1 -= int(diff/2)
            op1_y2 += int(diff/2)
        else :
            # higher than wide
            diff = op1_height - op1_width
            op1_x1 -= int(diff/2)
            op1_x2 += int(diff/2)


        op1_image = op1_image[op1_y1:op1_y2, op1_x1:op1_x2]
        cv2.imshow("op1_image_digits", op1_image)


        op2_image = self._canvas[self._operand2_top : self._operand2_top + self._operand2_height,  self._operand2_left : self._operand2_left + self._operand2_width]
        digits_op2 = digitdetector.detect(op2_image)
        print(digits_op2[0])
        op2_x1 = digits_op2[0][0]
        op2_y1 = digits_op2[0][1]
        op2_x2 = op2_x1 + digits_op2[0][2]
        op2_y2 = op2_y1 + digits_op2[0][3]
        op2_x1 -= op_pad
        op2_x2 += op_pad
        op2_y1 -= op_pad
        op2_y2 += op_pad
        op2_width = op2_x2 - op2_x1
        op2_height = op2_y2 - op2_y1
        if  (op2_width > op2_height):
            # wider than high
            diff = op2_width - op2_height
            op2_y1 -= int(diff/2)
            op2_y2 += int(diff/2)
        else :
            # higher than wide
            diff = op2_height - op2_width
            op2_x1 -= int(diff/2)
            op2_x2 += int(diff/2)

        op2_image = op2_image[op2_y1:op2_y2, op2_x1:op2_x2]
        cv2.imshow("op2_image_digits", op2_image)

        #cv2.imshow("op1", op1_image)
        #cv2.imshow("op2", op2_image)

        op1_labels, op1_probs = self.do_inference(self._graph, op1_image, 1)
        print ("op1 is: " + op1_labels[0] + "  op1 probability is: " + op1_probs[0])

        op2_labels, op2_probs = self.do_inference(self._graph, op2_image, 1)
        print ("op2 is: " + op2_labels[0] + "  op2 probability is: " + op2_probs[0])

        op1 = int(op1_labels[0])
        op2 = int(op2_labels[0])
        answer_int = op1 + op2
        self._answer_str = str(answer_int)




        # Draw the buttons again
        self._draw_ui()

        # Save the regions to file
        '''for n, rect in enumerate(digits):
            x, y, w, h = rect
            cv2.imwrite(str(n) + ".jpg", self._canvas[y:(y+h), x:(x+w)])'''

        # TODO: more stuff
        print('Submit!')


    def do_mvnc_initialize(self) -> (mvnc.Device, mvnc.Graph):
        """Creates and opens the Neural Compute device and
        creates a graph that can execute inferences on it.

        Returns
        -------
        device : mvnc.Device
            The opened device.  Will be None if couldn't open Device.
        graph : mvnc.Graph
            The allocated graph to use for inferences.  Will be None if couldn't allocate graph
        """
        # ***************************************************************
        # Get a list of ALL the sticks that are plugged in
        # ***************************************************************
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
                print('Error - No devices found')
                return (None, None)

        # ***************************************************************
        # Pick the first stick to run the network
        # ***************************************************************
        device = mvnc.Device(devices[0])

        # ***************************************************************
        # Open the NCS
        # ***************************************************************
        device.OpenDevice()

        filefolder = os.path.dirname(os.path.realpath(__file__))
        graph_filename = filefolder + '/mnist_inference.graph'

        # Load graph file
        try :
            with open(graph_filename, mode='rb') as f:
                in_memory_graph = f.read()
        except :
            print ("Error reading graph file: " + graph_filename)

        graph = device.AllocateGraph(in_memory_graph)

        return device, graph


    def do_mvnc_cleanup(self, device: mvnc.Device, graph: mvnc.Graph) -> None:
        """Cleans up the NCAPI resources.

        Parameters
        ----------
        device : mvncapi.Device
                 Device instance that was initialized in the do_initialize method
        graph : mvncapi.Graph
                Graph instance that was initialized in the do_initialize method

        Returns
        -------
        None

        """
        graph.DeallocateGraph()
        device.CloseDevice()


    def do_inference(self, graph: mvnc.Graph, input_image: str, number_results : int = 5) -> (List[str], List[numpy.float16]) :
        """ executes one inference which will determine the top classifications for an image file.

        Parameters
        ----------
        graph : Graph
            The graph to use for the inference.  This should be initialize prior to calling
        input_image : opencv image/Mat
            The image on which to run the inference.  if its not the right size will be resized internally
        number_results : int
            The number of results to return, defaults to 5

        Returns
        -------
        labels : List[str]
            The top labels for the inference.  labels[i] corresponds to probabilities[i]
        probabilities: List[numpy.float16]
            The top probabilities for the inference. probabilities[i] corresponds to labels[i]
        """

        # text labels for each of the possible classfications
        labels=[ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


        # Load image from disk and preprocess it to prepare it for the network
        # assuming we are reading a .jpg or .png color image so need to convert it
        # single channel gray scale image for mnist network.
        # Then resize the image to the size of image the network was trained with.
        # Next convert image to floating point format and normalize
        # so each pixel is a value between 0.0 and 1.0
        image_for_inference = cv2.bitwise_not(input_image)
        image_for_inference = cv2.cvtColor(image_for_inference, cv2.COLOR_BGR2GRAY)
        image_for_inference = cv2.resize(image_for_inference, self.NETWORK_IMAGE_DIMENSIONS)
        image_for_inference = image_for_inference.astype(numpy.float32)
        image_for_inference[:] = ((image_for_inference[:] )*(1.0/255.0))


        cv2.imshow("infer image_"+str(self._infer_count%2+1), image_for_inference)
        cv2.resizeWindow("infer image_"+str(self._infer_count%2+1), 100, 100)
        self._infer_count += 1

        # Start the inference by sending to the device/graph
        self._graph.LoadTensor(image_for_inference.astype(numpy.float16), None)

        # Get the result from the device/graph.  userobj should be the
        # same value that was passed in LoadTensor above.
        output, userobj = self._graph.GetResult()

        # sort indices in order of highest probabilities
        five_highest_indices = (-output).argsort()[:number_results]

        # get the labels and probabilities for the top results from the inference
        inference_labels = []
        inference_probabilities = []

        for index in range(0, number_results):
            inference_probabilities.append(str(output[five_highest_indices[index]]))
            inference_labels.append(labels[five_highest_indices[index]])

        return inference_labels, inference_probabilities



if __name__ == '__main__':
    touch_calc_window_title = "mnist calculator"
    app = TouchCalc(touch_calc_window_title)
    while True:

        if (app.is_window_closed()):
            break;

        app.show()

        key = cv2.waitKey(1)
        if key != -1:
            # Exit if any key is pressed
            break
    app.close()
