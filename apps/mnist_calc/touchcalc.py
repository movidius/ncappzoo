import cv2
import numpy as np

import digitdetector


class TouchCalc:
    def __init__(self, window_title='TouchCalc', width=800, height=800):
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
        cv2.setMouseCallback(self._window_name, self._draw)

        # The bottom 10% of the window will be a menu that can't be drawn on
        self._menu_bar_threshold = int(self._height * .9)

        # Set up a blank canvas
        self._canvas = np.zeros((width, height, 3), np.uint8)
        self._clear()

    def _clear(self):
        """Clear the canvas and redraw buttons."""
        self._canvas[:] = 255
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
                cv2.circle(self._canvas, (x, y), 10, (0, 0, 0), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False

    def _draw_ui(self, color=(255, 0, 0)):
        """Draw buttons on the canvas."""
        font_name = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2
        font_thickness = 3
        y_coord = self._height - 10

        cv2.putText(self._canvas, 'C', (0, y_coord), font_name, font_scale, color, font_thickness)
        cv2.putText(self._canvas, '=', (self._width - 50, y_coord), font_name, font_scale, color, font_thickness)

    def close(self):
        """Close and destroy the window."""
        cv2.destroyWindow(self._window_name)


    def is_window_closed(self):
        """ Determines if the user closed the window"""

        # check if the window has been closed.  all properties will return -1.0
        # for windows that are closed. If the user has closed the window via the
        # x on the title bar the property will be < 0 or an exception raised.  We are
        # getting property aspect ratio but it could probably be any property
        prop_asp = 1
        try:
            prop_asp = cv2.getWindowProperty(self._window_name, cv2.WND_PROP_ASPECT_RATIO)
        except:
            return True
        if (prop_asp < 0.0):
            # the property returned was < 0 so assume window was closed by user
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
        digits = digitdetector.detect(self._canvas)

        # Draw the buttons again
        self._draw_ui()

        # Save the regions to file
        '''for n, rect in enumerate(digits):
            x, y, w, h = rect
            cv2.imwrite(str(n) + ".jpg", self._canvas[y:(y+h), x:(x+w)])'''

        # TODO: more stuff
        print('Submit!')


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
