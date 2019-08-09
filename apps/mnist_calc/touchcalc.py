#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# Heather McCabe
# Touchscreen calculator app

import os

import cv2
import numpy as np

from mnist_processor import MnistProcessor
import calcui
from openvino.inference_engine import IENetwork, IEPlugin

class TouchCalc:
    def __init__(self, window_title='MNIST DrawCalc', width=1400, height=500):
        # Initialize mvncapi objects
        self._net_processor = None
        self._do_ncs_initialize()

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
        cv2.setMouseCallback(self._window_name, self._mouse_event)

        # Set up a blank white canvas
        self._canvas = np.zeros((height, width, 3), np.uint8)
        self._canvas[:] = 255

        # Sizes for calculating spacing
        operator_width = 80
        operator_height = 80
        padding = 10

        # Operands (the digits)
        self._operand1 = calcui.Operand(x=20, y=50, canvas=self._canvas, width=400, height=350, color=(230, 230, 230), thickness=1)
        self._operand2 = calcui.Operand(x=(self._operand1.x + self._operand1.width + operator_width + padding * 2),
                                 y=self._operand1.y,
                                 canvas=self._canvas,
                                 width=self._operand1.width,
                                 height=self._operand1.height,
                                 color=self._operand1.color,
                                 thickness=self._operand1.thickness)

        # Operators (+, -, *, /, etc.)
        operator_args = {'x': self._operand1.x + self._operand1.width + padding,
                         'y': self._operand1.y + int(self._operand1.height / 2) - int(operator_height / 2),
                         'canvas': self._canvas,
                         'width': operator_width,
                         'height': operator_height,
                         'thickness': 5}
        self._plus_sign = calcui.PlusSign(**operator_args)
        self._minus_sign = calcui.MinusSign(**operator_args)
        self._multiplication_sign = calcui.MultiplicationSign(**operator_args)
        self._division_sign = calcui.DivisionSign(**operator_args)

        # Set the default operator to +
        self._operator = self._plus_sign

        # Equals sign (=)
        self._equals_sign = calcui.EqualsSign(x=(self._operand2.x + self._operand2.width + padding),
                                       y=(self._operand2.y + int(self._operand2.height / 2) - int(operator_height / 2)),
                                       canvas=self._canvas,
                                       width=int(operator_width * 1.5),
                                       height=operator_height,
                                       thickness=5)

        # Clear button
        self._clear_all_button = calcui.Label(label='C', x=0, y=(self._height - int(self._height / 10)),
                                              canvas=self._canvas, thickness=3, scale=2)
        self._clear_all_button.x = self._width - self._clear_all_button.width

        # Operand labels
        self._op1_label = calcui.Label(label='', x=self._operand1.left, y=(self._operand1.bottom + 5),
                                       canvas=self._canvas, color=(255, 0, 0), thickness=2, scale=1)
        self._op2_label = calcui.Label(label='', x=self._operand2.left, y=(self._operand2.bottom + 5),
                                       canvas=self._canvas, color=(255, 0, 0), thickness=2, scale=1)

        # Answer label
        self._answer_label = calcui.Label(label='', x=self._equals_sign.right + 10, y=self._equals_sign.top,
                                          canvas=self._canvas, color=(255, 0, 0), thickness=3, scale=5)

        # Instruction label
        instructions = "Tap '=' to submit. Tap 'C' to clear. Tap the operator to change operations. Press any key to quit."
        self._instruction_label = calcui.Label(label=instructions, x=0, y=5, canvas=self._canvas, thickness=2, scale=0.85)

        # Calculation variables
        self._op1, self._op1_prob = None, None
        self._op2, self._op2_prob = None, None
        self._answer = None

        # Draw the UI
        self._draw_ui()

    def _clear_ui(self):
        """Clear the canvas and set detected digit values and the calculation answer to None."""
        self._canvas[:] = 255
        self._op1, self._op1_prob = None, None
        self._op2, self._op2_prob = None, None
        self._answer = None

    def _draw_ui(self):
        """Draw the UI elements."""
        self._operand1.draw()
        self._operand2.draw()
        self._operator.draw()
        self._equals_sign.draw()
        self._clear_all_button.draw()
        self._instruction_label.draw()

    def _draw_results(self):
        """Label the detected digits, their probabilities, and the answer."""
        # Clear old labels
        self._op1_label.clear()
        self._op2_label.clear()
        self._answer_label.clear()

        # Set label text... need to check 'if is not None' because if they are 0 they evaluate to False
        self._op1_label.label = '{:d} ({:.2f}% probability)'.format(self._op1, self._op1_prob * 100) if self._op1 is not None else 'No digit detected.'
        self._op2_label.label = '{:d} ({:.2f}% probability)'.format(self._op2, self._op2_prob * 100) if self._op2 is not None else 'No digit detected.'
        self._answer_label.label = str(self._answer) if self._answer is not None else None

        # Draw new labels
        self._op1_label.draw()
        self._op2_label.draw()
        self._answer_label.draw()

    def _mouse_event(self, event, x, y, flags, param):
        """Event listener for mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._equals_sign.contains_point(x, y):
                # Equal sign was clicked
                self.submit()
                self._draw_results()
            elif self._clear_all_button.contains_point(x, y):
                # Clear was clicked
                self._clear_ui()
                self._draw_ui()
            elif self._operator.contains_point(x, y):
                # The operator was clicked, swap to the next operator
                self._operator.clear()
                if self._operator is self._plus_sign:
                    self._operator = self._minus_sign
                elif self._operator is self._minus_sign:
                    self._operator = self._multiplication_sign
                elif self._operator is self._multiplication_sign:
                    self._operator = self._division_sign
                elif self._operator is self._division_sign:
                    self._operator = self._plus_sign
                self._operator.draw()
            else:
                self._drawing = True

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            if self._operand1.contains_point(x, y) or self._operand2.contains_point(x, y):
                # Draw if this is inside an operand rectangle
                if self._last_point:
                    cv2.line(self._canvas, self._last_point, (x, y), (0, 0, 0), 30)
                    self._last_point = (x, y)
                else:
                    self._last_point = (x, y)
            else:
                # Drawing outside the boundaries, forget last point so line won't connect when re-entering boundary
                self._last_point = None

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            self._last_point = None

    def _do_ncs_initialize(self):
        """Create and opens the Neural Compute device and create a MnistProcessor."""

        file_folder = os.path.dirname(os.path.realpath(__file__))
        graph_filename = file_folder + '/mnist_inference'
        
        self._plugin = IEPlugin(device="MYRIAD")
        # Create processor object for this network
        self._net_processor = MnistProcessor(graph_filename, self._plugin)

    def _do_ncs_cleanup(self):
        """Clean up the NCAPI resources."""
        del self._plugin
        self._net_processor.cleanup()

    def _do_ncs_infer(self, operand, img_label=None):
        """Detect and classify digits. If you provide an img_label the cropped digit image will be written to file."""
        # Get a list of rectangles for objects detected in this operand's box
        op_img = self._canvas[operand.top: operand.bottom, operand.left: operand.right]
        gray_img = cv2.cvtColor(op_img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digits = [cv2.boundingRect(contour) for contour in contours]

        if len(digits) > 0:
            x, y, w, h = digits[0]
            digit_img = self._canvas[operand.top + y: operand.top + y + h,
                                     operand.left + x: operand.left + x + w]

            # Write the cropped image to file if a label was provided
            if img_label:
                cv2.imwrite(img_label + ".png", digit_img)

            # Classify the digit and return the most probable result
            value, probability = self._net_processor.do_async_inference(digit_img)[0]
            print("value: " + str(value) + " probability: " + str(probability))
            return value, probability
        else:
            return None, None

    def close(self):
        """Close and destroy the window."""
        cv2.destroyWindow(self._window_name)
        self._do_ncs_cleanup()

    def is_window_closed(self):
        """Try to determine if the user closed the window (by clicking the x).

        This may only work with OpenCV 3.x.

        All OpenCV window properties should return -1.0 for windows that are closed.
        If we read a property that has a value < 0 or an exception is raised we assume
        the window has been closed. We use the aspect ratio property but it could be any.

        """
        try:
            prop_asp = cv2.getWindowProperty(self._window_name, cv2.WND_PROP_ASPECT_RATIO)
            if prop_asp < 0.0:
                # the property returned was < 0 so assume window was closed by user
                return True
        except:
            return True

        return False

    def show(self):
        """Show the window if hidden and update the display."""
        cv2.imshow(self._window_name, self._canvas)

    def submit(self):
        """Process the calculation when the submit button is clicked."""
        # Detect and classify digits
        self._op1, self._op1_prob = self._do_ncs_infer(self._operand1)
        self._op2, self._op2_prob = self._do_ncs_infer(self._operand2)

        # Calculate the answer (must do "is None" instead of "not" because 0 evaluates as False)
        if self._op1 is None or self._op2 is None:
            self._answer = None
        else:
            if self._operator is self._plus_sign:
                self._answer = self._op1 + self._op2
            elif self._operator is self._minus_sign:
                self._answer = self._op1 - self._op2
            elif self._operator is self._multiplication_sign:
                self._answer = self._op1 * self._op2
            elif self._operator is self._division_sign:
                # Will display "inf" if op2 is 0
                self._answer = self._op1 / self._op2


if __name__ == '__main__':
    app = TouchCalc('MNIST Calculator')
    while True:

        if cv2.waitKey(1) != -1 or app.is_window_closed():
            # Exit if any key is pressed or the window is closed
            break

        app.show()

    app.close()
