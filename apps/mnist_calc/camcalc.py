import cv2
import numpy as np
import queue
import threading

import digitdetector


class CamCalc:
    def __init__(self, window_title='CamCalc', width=1200, height=800):
        # Save these for use later
        self._height = height
        self._width = width

        # Create a window with OpenCV
        self._window_name = 'cam_window'
        self._window = cv2.namedWindow(self._window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self._window_name, self._width, self._height)
        cv2.setWindowTitle(self._window_name, window_title)

        # Set up a blank canvas
        self._blank = np.zeros((width, height, 3), np.uint8)

        # Set up the camera
        self._camera = cv2.VideoCapture(0)
        self._cam_thread = threading.Thread(target=self._capture)
        self._stop_event = threading.Event()
        self._exit_event = threading.Event()

        # Set up an input queue and single image frame
        self._queue = queue.Queue() # TODO: use Fifo class
        self._frame = self._blank

        # Start the camera
        self._run()

    def _capture(self):
        """Capture a frame from the camera and add it to the queue. This runs in the camera thread.."""
        while True:
            if self._stop_event.is_set():
                self._exit_event.set()
                break

            retVal, img = self._camera.read()
            self._queue.put(img)

    def _run(self):
        """Start the camera thread."""
        if not self._cam_thread.isAlive():
            self._cam_thread.setDaemon(True)
            self._stop_event.clear()
            self._exit_event.clear()
            self._queue.queue.clear()
            self._cam_thread.start()

    def _stop(self):
        """Stop the camera thread."""
        self._stop_event.set()
        self._queue.queue.clear()
        if self._exit_event.is_set():
            self._cam_thread.join()
        self._camera = None

    def close(self):
        """Close the application."""
        self._stop()
        cv2.destroyWindow(self._window_name)

    def show(self):
        """Show the window if hidden and update the display."""
        try:
            self._frame = self._queue.get()
        except queue.Empty:
            print('Could not retrieve camera frame.')
            self._frame = self._blank

        cv2.imshow(self._window_name, self._frame)

    def submit(self):
        """Process the image when the submit button is pressed."""
        # Detect the digits
        digits = digitdetector.detect(self._frame)

        # Save the regions to file
        '''for n, rect in enumerate(digits):
            x, y, w, h = rect
            cv2.imwrite(str(n) + ".jpg", self._frame[y:(y+h), x:(x+w)])'''

        # TODO: more stuff
        print('Calculate!')


if __name__ == '__main__':
    print('Press C to calculate. Press Q to exit.')
    app = CamCalc()
    while True:
        app.show()
        key = cv2.waitKey(1)
        if key == ord('c'):
            # Calculate
            app.submit()
        elif key == ord('q'):
            # Exit
            break
    app.close()
