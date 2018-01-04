To compile with NCSDK 1.11.00 replace the TensorFlowParser.py in the /usr/local/bin/ncsdk/Controllers directory with the TensorFlowParser.py.for_facenet file.

If just using the graph file then don't worry about the TensorFlowParser.py file, but you do need NCSDK 1.11.00 or above.

To run it it’s just python3 run.py from the facenet directory.
 
The file facenet/validated_images/valid.jpg contains the key image to which the images in the facenet directory are compared for a match.  It will bring up a window showing the test image with a red frame for no match or a green frame for a match.

You will likely need to play with the FACE_MATCH_THRESHOLD depending on your images.

If you want to use a webcam instead of images on the hard drive then change
use_camera = False
to 
use_camera = True 
in the main() function.

