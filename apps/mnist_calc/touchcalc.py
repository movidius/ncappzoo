#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# HMM & NPS

import os

import cv2
import mvnc.mvncapi as mvnc
import numpy as np

from mnist_processor import MnistProcessor


class UIElement:
    def __init__(self, x, y, width=1, height=1, color=(0, 0, 0), thickness=1):
        self.x = x
        self.y = y

        self._width = width
        self._height = height

        self.color = color
        self.thickness = thickness

    def __str__(self):
        return 'x: ' + str(self.x) + ', ' + \
               'y: ' + str(self.y) + ', ' + \
               'w: ' + str(self.width) + ', ' + \
               'h: ' + str(self.height) + ', ' + \
               'color: ' + str(self.color) + ', ' + \
               'thickness: ' + str(self.thickness)

    def contains_point(self, x, y):
        """Return True if a given point is within this element's boundaries, otherwise False."""
        if self.left <= x <= self.right and self.top <= y <= self.bottom:
            return True
        else:
            return False

    def clear(self, target):
        """Draw a filled white rectangle over this element to clear it from the canvas."""
        padding = self.thickness  # need to overwrite a slightly larger area or some drawn edges probably will remain
        cv2.rectangle(target, (self.left - padding, self.top - padding),
                      (self.right + padding, self.bottom + padding),
                      (255, 255, 255), cv2.FILLED)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.width

    @property
    def center_x(self):
        return self.x + int(self.width / 2)

    @property
    def center_y(self):
        return self.y + int(self.height / 2)


class Operand(UIElement):
    def draw(self, target):
        # Unfilled rectangle
        cv2.rectangle(target, (self.left, self.top), (self.right, self.bottom), self.color, self.thickness)


class PlusSign(UIElement):
    def draw(self, target):
        # Horizontal line
        cv2.line(target, (self.center_x, self.top), (self.center_x, self.bottom),
                 self.color, self.thickness)
        # Vertical line
        cv2.line(target, (self.left, self.center_y), (self.left + self.width, self.center_y),
                 self.color, self.thickness)


class MinusSign(UIElement):
    def draw(self, target):
        # Horizontal line
        cv2.line(target, (self.left, self.center_y), (self.right, self.center_y), self.color, self.thickness)


class MultiplicationSign(UIElement):
    def draw(self, target):
        # Crossed diagonal lines (X)
        cv2.line(target, (self.left, self.top), (self.right, self.bottom), self.color, self.thickness)
        cv2.line(target, (self.left, self.bottom), (self.right, self.top), self.color, self.thickness)


class DivisionSign(UIElement):
    def draw(self, target):
        # Horizontal line with dots above and below
        cv2.line(target, (self.left, self.center_y), (self.right, self.center_y), self.color, self.thickness)
        cv2.circle(target, (self.center_x, self.center_y - 30), 5, self.color, 5)
        cv2.circle(target, (self.center_x, self.center_y + 30), 5, self.color, 5)


class EqualsSign(UIElement):
    def draw(self, target):
        # Top line
        cv2.line(target, (self.left, self.center_y - 10), (self.center_x, self.center_y - 10),
                 self.color, self.thickness)
        # Bottom line
        cv2.line(target, (self.left, self.center_y + 10),(self.center_x, self.center_y + 10),
                 self.color, self.thickness)


class Label(UIElement):
    def __init__(self, x, y, label, color=(0, 0, 0), thickness=1, scale=1):
        super(Label, self).__init__(x, y, color=color, thickness=thickness)
        self.label = label
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.color = color
        self.scale = scale

    def clear(self, target):
        """Draw a filled white rectangle over this element to clear it from the canvas."""
        padding = self.thickness  # need to overwrite a slightly larger area or some drawn edges probably will remain
        cv2.rectangle(target, (self.left - padding, self.top - padding),
                      (self.right + padding, self.bottom + padding * 4),  # clear more beneath to handle hanging letters like 'g'
                      (255, 255, 255), cv2.FILLED)

    def draw(self, target):
        # Write text label
        cv2.putText(target, self.label, (self.left, self.bottom), self.font, self.scale, self.color, self.thickness)

    @property
    def width(self):
        size, baseline = cv2.getTextSize(self.label, self.font, self.scale, self.thickness)
        return size[0]

    @property
    def height(self):
        size, baseline = cv2.getTextSize(self.label, self.font, self.scale, self.thickness)
        return size[1]


class TouchCalc:
    def __init__(self, window_title='MNIST DrawCalc', width=1400, height=500):
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

        # Set up a blank canvas
        self._canvas = np.zeros((height, width, 3), np.uint8)

        # Sizes for calculating spacing
        operator_width = 80
        operator_height = 80
        padding = 10

        # Operands (the digits)
        self._operand1 = Operand(x=20, y=50, width=400, height=350, color=(230, 230, 230), thickness=1)
        self._operand2 = Operand(x=(self._operand1.x + self._operand1.width + operator_width + padding * 2),
                                 y=self._operand1.y,
                                 width=self._operand1.width,
                                 height=self._operand1.height,
                                 color=self._operand1.color,
                                 thickness=self._operand1.thickness)

        # Operators (+, -, *, /, etc.)
        operator_args = {'x': self._operand1.x + self._operand1.width + padding,
                         'y': self._operand1.y + int(self._operand1.height / 2) - int(operator_height / 2),
                         'width': operator_width,
                         'height': operator_height,
                         'thickness': 5}
        self._plus_sign = PlusSign(**operator_args)
        self._minus_sign = MinusSign(**operator_args)
        self._multiplication_sign = MultiplicationSign(**operator_args)
        self._division_sign = DivisionSign(**operator_args)

        # Set the default operator to +
        self._operator = self._plus_sign

        # Equals sign (=)
        self._equals_sign = EqualsSign(x=(self._operand2.x + self._operand2.width + padding),
                                       y=(self._operand2.y + int(self._operand2.height / 2) - int(operator_height / 2)),
                                       width=int(operator_width * 1.5),
                                       height=operator_height,
                                       thickness=5)

        # Clear button
        self._clear_all_button = Label(x=0, y=(self._height - int(self._height / 10)), label='C', scale=2, thickness=3)
        self._clear_all_button.x = self._width - self._clear_all_button.width

        # Operand labels
        self._op1_label = Label(x=self._operand1.left, y=(self._operand1.bottom + 5), label='',
                                scale=1, thickness=2, color=(255, 0, 0))
        self._op2_label = Label(x=self._operand2.left, y=(self._operand2.bottom + 5), label='',
                                scale=1, thickness=2, color=(255, 0, 0))

        # Answer label
        self._answer_label = Label(x=self._equals_sign.right + 10, y=self._equals_sign.top, label='',
                                   thickness=3, scale=5, color=(255, 0, 0))

        # Instruction label
        instructions = "Tap '=' to submit. Tap 'C' to clear. Tap the operator to change operations. Press any key to quit."
        self._instruction_label = Label(x=0, y=5, label=instructions, scale=0.85, thickness=2)

        # Calculation variables
        self._op1, self._op1_prob = None, None
        self._op2, self._op2_prob = None, None
        self._answer = None

        # Draw the screen
        self._draw_ui()

        # Initialize mvncapi objects
        self._device, self._net_processor = None, None
        self._do_mvnc_initialize()

    def _clear_ui(self):
        """Clear the digits and answer and redraw the UI."""
        self._op1, self._op1_prob = None, None
        self._op2, self._op2_prob = None, None
        self._answer = None
        self._draw_ui()

    def _draw_ui(self):
        """Clear the UI and draw the UI elements."""
        # Clear the canvas
        self._canvas[:] = 255

        # Draw UI elements
        self._operand1.draw(self._canvas)
        self._operand2.draw(self._canvas)
        self._operator.draw(self._canvas)
        self._equals_sign.draw(self._canvas)
        self._clear_all_button.draw(self._canvas)
        self._instruction_label.draw(self._canvas)

    def _draw_results(self):
        """Label the detected digits, their probabilities, and the answer."""
        # Clear old labels
        self._op1_label.clear(self._canvas)
        self._op2_label.clear(self._canvas)
        self._answer_label.clear(self._canvas)

        # Set label text... need to check 'if is not None' because if they are 0 they evaluate to False
        self._op1_label.label = '{:d} ({:.2f}% probability)'.format(self._op1, self._op1_prob * 100) if self._op1 is not None else 'No digit detected.'
        self._op2_label.label = '{:d} ({:.2f}% probability)'.format(self._op2, self._op2_prob * 100) if self._op2 is not None else 'No digit detected.'
        self._answer_label.label = str(self._answer) if self._answer is not None else None

        # Draw new labels
        self._op1_label.draw(self._canvas)
        self._op2_label.draw(self._canvas)
        self._answer_label.draw(self._canvas)

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
            elif self._operator.contains_point(x, y):
                # The operator was clicked, swap to the next operator
                self._operator.clear(self._canvas)
                if self._operator is self._plus_sign:
                    self._operator = self._minus_sign
                elif self._operator is self._minus_sign:
                    self._operator = self._multiplication_sign
                elif self._operator is self._multiplication_sign:
                    self._operator = self._division_sign
                elif self._operator is self._division_sign:
                    self._operator = self._plus_sign
                self._operator.draw(self._canvas)
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

    def _do_mvnc_initialize(self):
        """Create and opens the Neural Compute device and create a MnistProcessor."""
        # Get a list of all devices
        devices = mvnc.enumerate_devices()
        if len(devices) == 0:
            print('Error - No devices found')
            return

        # Use the first device to run the network
        self._device = mvnc.Device(devices[0])
        self._device.open()

        file_folder = os.path.dirname(os.path.realpath(__file__))
        graph_filename = file_folder + '/mnist_inference.graph'

        # Create processor object for this network
        self._net_processor = MnistProcessor(graph_filename, self._device)

    def _do_mvnc_cleanup(self):
        """Clean up the NCAPI resources."""
        self._net_processor.cleanup()
        self._device.close()
        self._device.destroy()

    def _do_mvnc_infer(self, operand, img_label=None):
        """Detect and classify digits. If you provide an img_label the cropped digit image will be written to file."""
        # Get a list of rectangles for objects detected in this operand's box
        op_img = self._canvas[operand.top: operand.bottom, operand.left: operand.right]
        gray_img = cv2.cvtColor(op_img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
        _, contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digits = [cv2.boundingRect(contour) for contour in contours]

        if len(digits) > 0:
            x, y, w, h = digits[0]
            digit_img = self._canvas[operand.top + y: operand.top + y + h,
                                     operand.left + x: operand.left + x + w]

            # Write the cropped image to file if a label was provided
            if img_label:
                cv2.imwrite(img_label + ".png", digit_img)

            # Classify the digit and return the most probable result
            value, probability = self._net_processor.do_sync_inference(digit_img)[0]
            return value, probability
        else:
            return None, None

    def close(self):
        """Close and destroy the window."""
        cv2.destroyWindow(self._window_name)
        self._do_mvnc_cleanup()

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
        self._op1, self._op1_prob = self._do_mvnc_infer(self._operand1, 'op1')
        self._op2, self._op2_prob = self._do_mvnc_infer(self._operand2, 'op2')

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
