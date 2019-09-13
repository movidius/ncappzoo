
class Face_detection_result():
    '''
    Used to hold various face information.
    
    Parameters
    ----------
    name (str) - Name of the person.
    image_path (str) - File path of the image of the person.
    image_mat (OpenCV Mat) - OpenCV Mat of the image of the person.
    feature_vector (List of 512 floats) - List of 512 dimensional embeddings.
    face_mat (OpenCV Mat) - OpenCV Mat of the person's face.
    box_left (int) - Left side of the bounding box for the face.
    box_right (int) - Right side of the bounding box for the face.
    box_top (int) - Top side of the bounding box for the face.
    box_bottom (int) - Bottom side of the bounding box for the face.
    '''
    def __init__(self, face_mat=None, box_left=None, box_right=None, box_top=None, box_bottom=None):
        self.face_mat = face_mat
        self.box_left = box_left
        self.box_right = box_right
        self.box_top = box_top
        self.box_bottom = box_bottom
