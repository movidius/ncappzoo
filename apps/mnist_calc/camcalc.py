import cv2
import numpy as np
import queue

from cameraprocessor import CameraProcessor
import digitdetector


class CamCalc:
    def __init__(self, window_title='CamCalc', width=1200, height=800):
        # Create a window with OpenCV
        self._window_name = 'cam_window'
        self._window = cv2.namedWindow(self._window_name, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self._window_name, width, height)
        cv2.setWindowTitle(self._window_name, window_title)

        # Set up a blank canvas
        self._blank = np.zeros((width, height, 3), np.uint8)

        # Set up an input queue and single image frame
        self._queue = queue.Queue() # TODO: use Fifo class?
        self._frame = self._blank

        # Set up the camera
        self._camera = CameraProcessor(self._queue, request_video_width=width, request_video_height=height)

        # Start the camera
        self._camera.start_processing()

    def close(self):
        """Stop the camera thread and close the application."""
        self._camera.stop_processing()
        self._camera.cleanup()
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
